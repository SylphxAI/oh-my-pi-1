import { expect, it } from "bun:test";
import * as http2 from "node:http2";
import { streamCursor } from "@oh-my-pi/pi-ai/providers/cursor";
import type { ToolResultMessage, UserMessage } from "@oh-my-pi/pi-ai/types";
import { LiveSteeringChannel } from "../../agent/src/live-steering";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import {
	AgentServerMessageSchema,
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
import { create, toBinary } from "@oh-my-pi/pi-catalog/discovery/protobuf";

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
