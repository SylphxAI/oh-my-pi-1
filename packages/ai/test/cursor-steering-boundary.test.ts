import { expect, it } from "bun:test";
import * as http2 from "node:http2";
import { streamCursor } from "@oh-my-pi/pi-ai/providers/cursor";
import { Agent } from "../../agent/src/agent";
import type { Message } from "../src/types";
import { createMockModel } from "../src/providers/mock";
import type { ToolResultMessage, UserMessage } from "@oh-my-pi/pi-ai/types";
import { LiveSteeringChannel } from "../../agent/src/live-steering";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import {
	AgentClientMessageSchema,
	AgentServerMessageSchema,
	ConversationStateStructureSchema,
	FetchArgsSchema,
	InteractionQuerySchema,
	WebFetchRequestQuerySchema,
	ConnectScmArgsSchema,
	ConnectScmToolCallSchema,
	ToolCallSchema,
	ToolCallStartedUpdateSchema,
	ToolCallCompletedUpdateSchema,
	ExecServerMessageSchema,
	InteractionUpdateSchema,
	ReadArgsSchema,
	TurnEndedUpdateSchema,
} from "@oh-my-pi/pi-catalog/discovery/cursor-proto";
import { create, fromBinary, toBinary } from "@oh-my-pi/pi-catalog/discovery/protobuf";

function frame(data: Uint8Array): Buffer {
	const result = Buffer.alloc(5 + data.length);
	result.writeUInt32BE(data.length, 1);
	result.set(data, 5);
	return result;
}

it.each(["text", "image", "hosted"])("yields a five-tool Cursor run at step two with %s steering", async mode => {
	const sessions = new Set<http2.ServerHttp2Session>();
	const server = http2.createServer();
	server.on("session", session => {
		sessions.add(session);
		session.on("close", () => sessions.delete(session));
	});
	server.on("stream", (socket: http2.ServerHttp2Stream) => {
		socket.respond({ ":status": 200, "content-type": "application/connect+proto" });
		let step = 0;
		socket.on("data", () => {
			step++;
			if (mode === "hosted" && step === 2) {
				const toolCall = create(ToolCallSchema, {
					tool: {
						case: "connectScmToolCall",
						value: create(ConnectScmToolCallSchema, {
							args: create(ConnectScmArgsSchema, { toolCallId: "call-2" }),
						}),
					},
				});
				const updates = [
					create(InteractionUpdateSchema, {
						message: {
							case: "toolCallStarted",
							value: create(ToolCallStartedUpdateSchema, { callId: "call-2", toolCall }),
						},
					}),
					create(InteractionUpdateSchema, {
						message: {
							case: "toolCallCompleted",
							value: create(ToolCallCompletedUpdateSchema, { callId: "call-2", toolCall }),
						},
					}),
				];
				const nextExec = create(AgentServerMessageSchema, {
					message: {
						case: "execServerMessage",
						value: create(ExecServerMessageSchema, {
							id: 3,
							execId: "exec-3",
							message: {
								case: "readArgs",
								value: create(ReadArgsSchema, { path: "/step-3", toolCallId: "call-3" }),
							},
						}),
					},
				});
				socket.write(
					Buffer.concat([
						...updates.map(value =>
							frame(
								toBinary(
									AgentServerMessageSchema,
									create(AgentServerMessageSchema, {
										message: { case: "interactionUpdate", value },
									}),
								),
							),
						),
						frame(toBinary(AgentServerMessageSchema, nextExec)),
					]),
				);
				step = 3;
				return;
			}
			const message =
				step <= 5
					? create(AgentServerMessageSchema, {
							message: {
								case: "execServerMessage",
								value: create(ExecServerMessageSchema, {
									id: step,
									execId: `exec-${step}`,
									message: {
										case: "readArgs",
										value: create(ReadArgsSchema, { path: `/step-${step}`, toolCallId: `call-${step}` }),
									},
								}),
							},
						})
					: create(AgentServerMessageSchema, {
							message: {
								case: "interactionUpdate",
								value: create(InteractionUpdateSchema, {
									message: { case: "turnEnded", value: create(TurnEndedUpdateSchema, {}) },
								}),
							},
						});
			socket.write(frame(toBinary(AgentServerMessageSchema, message)));
			if (step > 5) socket.end();
		});
	});
	const listening = Promise.withResolvers<void>();
	server.once("listening", listening.resolve);
	server.listen(0, "127.0.0.1");
	await listening.promise;
	const address = server.address();
	if (!address || typeof address === "string") throw new Error("missing fixture address");
	const model = buildModel({
		id: "cursor-steering-fixture",
		name: "Cursor steering fixture",
		api: "cursor-agent",
		provider: "cursor",
		baseUrl: `http://127.0.0.1:${address.port}`,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	});
	const executed: string[] = [];
	const results: ToolResultMessage[] = [];
	let queued = false;
	const steeringMessage: UserMessage = {
		role: "user",
		content:
			mode === "image" ? [{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }] : "Telegram steer at step two",
		timestamp: 2,
	};
	const steering = new LiveSteeringChannel({
		wait: async () => {},
		take: async () => {
			if (!queued) return [];
			queued = false;
			return [steeringMessage];
		},
		toProvider: async () => {
			if (mode === "hosted") {
				const conversion = Promise.withResolvers<void>();
				setImmediate(conversion.resolve);
				await conversion.promise;
			}
			return mode === "image" ? undefined : [steeringMessage];
		},
	});
	try {
		const response = streamCursor(
			model,
			{ messages: [{ role: "user", content: "five steps", timestamp: 1 }] },
			{
				apiKey: "test-token",
				transport: "http2",
				signal: AbortSignal.timeout(2000),
				liveSteering: steering,
				execHandlers: {
					read: async args => {
						executed.push(args.path);
						if (executed.length === 2) queued = true;
						return {
							role: "toolResult",
							toolCallId: args.toolCallId,
							toolName: "read",
							content: [{ type: "text", text: args.path }],
							isError: false,
							timestamp: 1,
						};
					},
				},
				onToolResult: async result => {
					results.push(result);
					if (mode === "hosted" && result.toolCallId === "call-2") queued = true;
					return result;
				},
			},
		);
		const result = await response.result();
		expect(result.stopReason).toBe("toolUse");
		expect(executed).toEqual(mode === "hosted" ? ["/step-1:raw"] : ["/step-1:raw", "/step-2:raw"]);
		expect(results.map(result => result.toolCallId)).toEqual(["call-1", "call-2"]);
		expect(steering.deferred).toEqual([steeringMessage]);
		expect(steering.accepted).toEqual([]);
	} finally {
		for (const session of sessions) session.destroy();
		server.close();
	}
});

it.each([
	"ended",
	"interrupted",
	"ordered",
	"query",
	"local-exec",
	"local-query",
	"fixed-exec",
	"fixed-query",
	"late-abort",
	"ended-abort",
	"interrupted-abort",
	"interrupted-checkpoint",
	"ended-hook-abort",
	"interrupted-hook-abort",
])("settles delayed hosted steering before %s finalization", async mode => {
	const interrupted = mode.startsWith("interrupted");
	const abort = mode.endsWith("abort");
	const holdHook = mode.includes("hook");
	const controller = new AbortController();
	let approvals = 0;
	let runs = 0;
	const local = mode.startsWith("local");
	const fixed = mode.startsWith("fixed");
	const firstHookStarted = Promise.withResolvers<void>();
	const releaseFirstHook = Promise.withResolvers<void>();
	const sink: string[] = [];
	const executed: string[] = [];
	const sessions = new Set<http2.ServerHttp2Session>();
	const server = http2.createServer();
	server.on("session", session => {
		sessions.add(session);
		session.on("close", () => sessions.delete(session));
	});
	const conversionStarted = Promise.withResolvers<void>();
	const releaseConversion = Promise.withResolvers<void>();
	const releaseLaterConversion = Promise.withResolvers<void>();
	const pending: UserMessage[] = [];
	const converted: string[] = [];
	const steering = new LiveSteeringChannel({
		wait: async () => {},
		take: async () => pending.splice(0, 1),
		toProvider: async messages => {
			const message = messages[0] as UserMessage;
			converted.push(String(message.content));
			if (message.content === "A") {
				conversionStarted.resolve();
				await releaseConversion.promise;
			} else {
				await releaseLaterConversion.promise;
			}
			return messages as UserMessage[];
		},
	});
	server.on("stream", (socket: http2.ServerHttp2Stream) => {
		const continuation = ++runs > 1;
		socket.respond({ ":status": 200, "content-type": "application/connect+proto" });
		socket.on("data", (chunk: Buffer) => {
			// Decode client replies separately from the first run request.
			let offset = 0;
			while (offset + 5 <= chunk.length) {
				const length = chunk.readUInt32BE(offset + 1);
				if (offset + 5 + length > chunk.length) break;
				const message = fromBinary(AgentClientMessageSchema, chunk.subarray(offset + 5, offset + 5 + length));
				if (message.message.case === "interactionResponse") approvals++;
				offset += 5 + length;
			}
		});
		socket.once("data", async () => {
			const frames: Buffer[] = [];
			if (local) {
				socket.write(
					frame(
						toBinary(
							AgentServerMessageSchema,
							create(AgentServerMessageSchema, {
								message: {
									case: "execServerMessage",
									value: create(ExecServerMessageSchema, {
										id: 1,
										execId: "local-A",
										message: {
											case: "readArgs",
											value: create(ReadArgsSchema, { path: "/A", toolCallId: "A" }),
										},
									}),
								},
							}),
						),
					),
				);
				await conversionStarted.promise;
			}
			for (const callId of continuation
				? ["C"]
				: fixed
					? ["A", "C"]
					: local
						? []
						: mode === "ordered" || mode === "late-abort"
							? ["A", "B"]
							: ["A"]) {
				const toolCall = create(ToolCallSchema, {
					tool: {
						case: "connectScmToolCall",
						value: create(ConnectScmToolCallSchema, {
							args: create(ConnectScmArgsSchema, { toolCallId: callId }),
						}),
					},
				});
				const updates = [
					create(InteractionUpdateSchema, {
						message: {
							case: "toolCallStarted",
							value: create(ToolCallStartedUpdateSchema, { callId, toolCall }),
						},
					}),
				];
				if (
					continuation ||
					fixed ||
					mode === "ordered" ||
					mode === "query" ||
					mode === "late-abort" ||
					mode === "interrupted-checkpoint"
				)
					updates.push(
						create(InteractionUpdateSchema, {
							message: {
								case: "toolCallCompleted",
								value: create(ToolCallCompletedUpdateSchema, { callId, toolCall }),
							},
						}),
					);
				for (const value of updates)
					frames.push(
						frame(
							toBinary(
								AgentServerMessageSchema,
								create(AgentServerMessageSchema, { message: { case: "interactionUpdate", value } }),
							),
						),
					);
			}
			if (mode === "local-exec" || mode === "fixed-exec") {
				const boundary = frame(
					toBinary(
						AgentServerMessageSchema,
						create(AgentServerMessageSchema, {
							message: {
								case: "execServerMessage",
								value: create(ExecServerMessageSchema, {
									id: 2,
									execId: "local-B",
									message: {
										case: "readArgs",
										value: create(ReadArgsSchema, { path: "/B", toolCallId: "B" }),
									},
								}),
							},
						}),
					),
				);
				if (fixed) frames.splice(2, 0, boundary);
				else frames.push(boundary);
				socket.write(Buffer.concat(frames));
				return;
			}
			if (mode === "query" || mode === "local-query" || mode === "fixed-query") {
				const boundary = frame(
					toBinary(
						AgentServerMessageSchema,
						create(AgentServerMessageSchema, {
							message: {
								case: "interactionQuery",
								value: create(InteractionQuerySchema, {
									id: 18,
									query: {
										case: "webFetchRequestQuery",
										value: create(WebFetchRequestQuerySchema, {
											args: create(FetchArgsSchema, {
												url: "https://example.com",
												toolCallId: "fetch-next",
											}),
										}),
									},
								}),
							},
						}),
					),
				);
				if (fixed) frames.splice(2, 0, boundary);
				else frames.push(boundary);
				socket.write(Buffer.concat(frames));
				return;
			}
			if (mode === "interrupted-checkpoint")
				frames.push(
					frame(
						toBinary(
							AgentServerMessageSchema,
							create(AgentServerMessageSchema, {
								message: {
									case: "conversationCheckpointUpdate",
									value: create(ConversationStateStructureSchema, {
										rootPromptMessagesJson: [Uint8Array.of(1)],
										turns: [Uint8Array.of(2)],
									}),
								},
							}),
						),
					),
				);
			if (!interrupted)
				frames.push(
					frame(
						toBinary(
							AgentServerMessageSchema,
							create(AgentServerMessageSchema, {
								message: {
									case: "interactionUpdate",
									value: create(InteractionUpdateSchema, {
										message: { case: "turnEnded", value: create(TurnEndedUpdateSchema, {}) },
									}),
								},
							}),
						),
					),
				);
			socket.end(Buffer.concat(frames));
		});
	});
	const listening = Promise.withResolvers<void>();
	server.once("listening", listening.resolve);
	server.listen(0, "127.0.0.1");
	await listening.promise;
	const address = server.address();
	if (!address || typeof address === "string") throw new Error("missing fixture address");
	const model = buildModel({
		id: "cursor-finalization-fixture",
		name: "Cursor finalization fixture",
		api: "cursor-agent",
		provider: "cursor",
		baseUrl: `http://127.0.0.1:${address.port}`,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	});
	try {
		let finished = false;
		const result = streamCursor(
			model,
			{ messages: [{ role: "user", content: "hosted tools", timestamp: 1 }] },
			{
				apiKey: "test-token",
				transport: "http2",
				signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]),
				providerRetryWait: async () => {},
				liveSteering: steering,
				execHandlers: {
					read: async args => {
						executed.push(args.toolCallId);
						return {
							role: "toolResult",
							toolCallId: args.toolCallId,
							toolName: "read",
							content: [{ type: "text", text: args.path }],
							isError: false,
							timestamp: 1,
						};
					},
				},
				onToolResult: async toolResult => {
					sink.push(toolResult.toolCallId);
					if (fixed && toolResult.toolCallId === "A") {
						firstHookStarted.resolve();
						await releaseFirstHook.promise;
						return toolResult;
					}
					if (fixed && toolResult.toolCallId !== "C") return toolResult;
					if (holdHook) {
						conversionStarted.resolve();
						await releaseConversion.promise;
					}
					pending.push({ role: "user", content: fixed ? "A" : toolResult.toolCallId, timestamp: 2 });
					return toolResult;
				},
			},
		)
			.result()
			.then(message => {
				finished = true;
				return message;
			});
		if (fixed) {
			await firstHookStarted.promise;
			releaseFirstHook.resolve();
		}
		await conversionStarted.promise;
		if (abort) {
			controller.abort();
			const checkpoint = Promise.withResolvers<void>();
			setImmediate(checkpoint.resolve);
			await checkpoint.promise;
			expect(finished).toBe(true);
			expect((await result).stopReason).toBe("aborted");
			if (mode === "late-abort") {
				const snapshot = [...steering.deferred];
				expect(snapshot.map(message => message.content)).toEqual(["A"]);
				expect(pending).toEqual([]);
				const next = await streamCursor(
					model,
					{ messages: snapshot as UserMessage[] },
					{
						apiKey: "test-token",
						transport: "http2",
						signal: AbortSignal.timeout(2000),
						onToolResult: toolResult => {
							sink.push(toolResult.toolCallId);
							return toolResult;
						},
					},
				).result();
				expect(next.stopReason).toBe("stop");
				expect(sink).toEqual(["A", "C"]);
				releaseConversion.resolve();
				releaseLaterConversion.resolve();
				const settled = Promise.withResolvers<void>();
				setImmediate(settled.resolve);
				await settled.promise;
				expect(sink).toEqual(["A", "C"]);
				expect(steering.deferred).toEqual(snapshot);
			}
			return;
		}
		if (mode === "query" || local || fixed) {
			// A transport round trip places our observation after the client has
			// processed the chunk containing both completion and permission query.
			const roundTrip = Promise.withResolvers<void>();
			for (const session of sessions) session.ping(error => (error ? roundTrip.reject(error) : roundTrip.resolve()));
			await roundTrip.promise;
		}
		// B is allowed to finish first; it must still wait behind A's settlement.
		releaseLaterConversion.resolve();
		// Let all callbacks and stream finalization reach the explicit conversion gate.
		const checkpoint = Promise.withResolvers<void>();
		setImmediate(checkpoint.resolve);
		await checkpoint.promise;
		expect(finished).toBe(false);
		expect(approvals).toBe(0);
		expect(executed).toEqual(local ? ["A"] : []);
		expect(converted).toEqual(["A"]);
		expect(steering.deferred).toEqual([]);
		releaseConversion.resolve();
		const message = await result;
		expect(message.stopReason).toBe("toolUse");
		expect([...steering.deferred, ...pending].map(message => message.content)).toEqual(
			mode === "ordered" ? ["A", "B"] : ["A"],
		);
		expect(steering.accepted).toEqual([]);
		expect(approvals).toBe(0);
		if (mode === "interrupted-checkpoint") expect(runs).toBe(1);
	} finally {
		releaseFirstHook.resolve();
		releaseConversion.resolve();
		releaseLaterConversion.resolve();
		for (const session of sessions) session.destroy();
		server.close();
	}
});

it("pairs an open hosted call on cancellation in the agent transcript and continuation", async () => {
	const sessions = new Set<http2.ServerHttp2Session>();
	const server = http2.createServer();
	server.on("session", session => {
		sessions.add(session);
		session.on("close", () => sessions.delete(session));
	});
	server.on("stream", (socket: http2.ServerHttp2Stream) => {
		socket.respond({ ":status": 200, "content-type": "application/connect+proto" });
		socket.once("data", () => {
			const toolCall = create(ToolCallSchema, {
				tool: {
					case: "connectScmToolCall",
					value: create(ConnectScmToolCallSchema, {
						args: create(ConnectScmArgsSchema, { toolCallId: "open-hosted" }),
					}),
				},
			});
			socket.write(
				frame(
					toBinary(
						AgentServerMessageSchema,
						create(AgentServerMessageSchema, {
							message: {
								case: "interactionUpdate",
								value: create(InteractionUpdateSchema, {
									message: {
										case: "toolCallStarted",
										value: create(ToolCallStartedUpdateSchema, {
											callId: "open-hosted",
											toolCall,
										}),
									},
								}),
							},
						}),
					),
				),
			);
		});
	});
	const listening = Promise.withResolvers<void>();
	server.once("listening", listening.resolve);
	server.listen(0, "127.0.0.1");
	await listening.promise;
	const address = server.address();
	if (!address || typeof address === "string") throw new Error("missing fixture address");
	const model = buildModel({
		id: "cursor-abort-pairing-fixture",
		name: "Cursor abort pairing fixture",
		api: "cursor-agent",
		provider: "cursor",
		baseUrl: `http://127.0.0.1:${address.port}`,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	});
	const opened = Promise.withResolvers<void>();
	const mock = createMockModel({ responses: [{ content: ["continued"] }] });
	let continuation: Message[] = [];
	let calls = 0;
	const agent = new Agent({
		initialState: { model },
		streamFn: (model, context, options) => {
			if (++calls > 1) {
				continuation = context.messages;
				return mock.stream(mock.model, context, options);
			}
			return streamCursor(model, context, {
				...options,
				onToolResult: options?.cursorOnToolResult,
				apiKey: "test-token",
				transport: "http2",
			});
		},
	});
	agent.subscribe(event => {
		if (event.type === "message_update" && event.assistantMessageEvent.type === "toolcall_start") opened.resolve();
	});
	const running = agent.prompt("start hosted call");
	try {
		await opened.promise;
		agent.abort();
		await running;
		const paired = agent.state.messages.filter(message => message.role === "toolResult");
		expect(paired.map(message => message.toolCallId)).toEqual(["open-hosted"]);
		expect(paired[0]?.isError).toBe(true);
		await agent.continue();
		expect(
			continuation.flatMap(message =>
				message.role === "assistant"
					? message.content.flatMap(block => (block.type === "toolCall" ? [block.id] : []))
					: [],
			),
		).toEqual(["open-hosted"]);
		expect(continuation.flatMap(message => (message.role === "toolResult" ? [message.toolCallId] : []))).toEqual([
			"open-hosted",
		]);
	} finally {
		agent.abort();
		await running;
		for (const session of sessions) session.destroy();
		server.close();
	}
});
