import * as http2 from "node:http2";
import { streamCursor } from "@oh-my-pi/pi-ai/providers/cursor";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import {
	AgentServerMessageSchema,
	ConnectScmArgsSchema,
	ConnectScmToolCallSchema,
	ToolCallSchema,
	InteractionUpdateSchema,
	ToolCallStartedUpdateSchema,
	ToolCallCompletedUpdateSchema,
} from "@oh-my-pi/pi-catalog/discovery/cursor-proto";
import { create, toBinary } from "@oh-my-pi/pi-catalog/discovery/protobuf";
import { describe, expect, it } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { agentLoop } from "@oh-my-pi/pi-agent-core/agent-loop";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolContext,
	StreamFn,
} from "@oh-my-pi/pi-agent-core/types";
import type { LiveSteering, Message } from "@oh-my-pi/pi-ai";
import { createMockModel, type MockResponse } from "@oh-my-pi/pi-ai/providers/mock";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { CodexSteerPump, type CodexSteerAck } from "../../ai/src/providers/openai-codex/live-steering";
import { LiveSteeringChannel } from "../src/live-steering";
import { createUserMessage } from "./helpers";

/** Marks user text so tests can see the provider received the converted view. */
function convertToLlm(messages: AgentMessage[]): Message[] {
	const out: Message[] = [];
	for (const message of messages) {
		if (message.role === "user") {
			out.push({ ...message, content: `<user>${message.content}</user>` });
		} else if (message.role === "assistant" || message.role === "toolResult") {
			out.push(message);
		} else if (message.role === "custom") {
			out.push({ role: "developer", content: String(message.content), timestamp: message.timestamp });
		}
	}
	return out;
}

function textOf(message: Message): string {
	if (message.role === "assistant") {
		return message.content.map(block => (block.type === "text" ? block.text : "")).join("");
	}
	return typeof message.content === "string" ? message.content : "";
}

interface SteeredRunOptions {
	/** Steering already queued when the first provider call opens. */
	initialQueue?: AgentMessage[];
	/** First response; later ones are plain text. */
	first?: MockResponse;
	tools?: AgentTool[];
	getToolContext?: AgentLoopConfig["getToolContext"];
}

/**
 * Runs a three-response conversation whose first provider call runs `steer`
 * against the live-steering source it was offered; returns the text of every
 * provider call's context.
 */
async function runSteered(
	steer: (live: LiveSteering, queue: AgentMessage[]) => Promise<void>,
	setup: SteeredRunOptions = {},
): Promise<{ contexts: string[][]; transcript: AgentMessage[]; events: AgentEvent[] }> {
	const queue: AgentMessage[] = [];
	const mock = createMockModel({
		responses: [setup.first ?? { content: ["first"] }, { content: ["second"] }, { content: ["third"] }],
	});
	const contexts: string[][] = [];
	const streamFn: StreamFn = async (model, context, options) => {
		contexts.push(context.messages.map(textOf));
		if (contexts.length === 1) {
			if (!options?.liveSteering) throw new Error("live steering was not offered");
			queue.push(...(setup.initialQueue ?? []));
			await steer(options.liveSteering, queue);
		}
		return mock.stream(model, context, options);
	};
	const config: AgentLoopConfig = {
		model: mock.model,
		convertToLlm,
		getSteeringMessages: async () => queue.splice(0),
		waitForSteeringMessages: async () => {},
		getToolContext: setup.getToolContext,
	};
	const context: AgentContext = { systemPrompt: [""], messages: [], tools: setup.tools ?? [] };
	const run = agentLoop([createUserMessage("start")], context, config, undefined, streamFn);
	const events: AgentEvent[] = [];
	for await (const event of run) events.push(event);
	return { contexts, transcript: await run.result(), events };
}

/**
 * Starts an `Agent` run whose provider stream stays open until aborted and
 * claims one steer live without settling it, as while `response.steer` awaits
 * its acknowledgement; resolves once claimed.
 */
async function startLiveSteeredRun(): Promise<{ agent: Agent; steer: AgentMessage; running: Promise<void> }> {
	const claimed = Promise.withResolvers<void>();
	const agent = new Agent({
		streamFn: async (_model, _context, options) => {
			const live = options?.liveSteering;
			const signal = options?.signal;
			if (!live || !signal) throw new Error("live steering was not offered");
			const stream = new AssistantMessageEventStream();
			signal.addEventListener("abort", () => stream.fail(new Error("aborted")), { once: true });
			await live.wait(signal);
			if (!(await live.claim(signal))) throw new Error("steer was not claimed");
			claimed.resolve();
			return stream;
		},
	});
	const running = agent.prompt("start");
	const steer = createUserMessage("use tabs");
	agent.steer(steer);
	await claimed.promise;
	return { agent, steer, running };
}

/** Transcript user messages the UI marks as delivered live. */
function liveSteeredTexts(transcript: AgentMessage[]): unknown[] {
	const texts: unknown[] = [];
	for (const message of transcript) {
		if (message.role === "user" && message.liveSteered) texts.push(message.content);
	}
	return texts;
}

function userText(message: AgentMessage): string {
	if (message.role !== "user") return "";
	const { content } = message;
	return typeof content === "string" ? content : content.map(part => (part.type === "text" ? part.text : "")).join("");
}

describe("agent loop live steering", () => {
	it("records a submitted Codex steer accepted after finish stops claiming", async () => {
		const steer = createUserMessage("use tabs");
		const queue = [steer];
		const channel = new LiveSteeringChannel({
			wait: async () => {},
			take: async () => queue.splice(0),
			toProvider: async () => [steer],
		});
		const submitted = Promise.withResolvers<void>();
		const acknowledgement = Promise.withResolvers<CodexSteerAck>();
		const pump = new CodexSteerPump(
			channel,
			{
				steer: async () => {
					submitted.resolve();
					return acknowledgement.promise;
				},
			},
			() => [{ role: "user", content: "use tabs" }],
		);
		pump.start("response-1");
		await submitted.promise;
		const finishing = pump.finish();
		try {
			expect(channel.deferred).toEqual([]);
		} finally {
			acknowledgement.resolve({ accepted: true, id: "steer-1" });
			await finishing;
		}
		const outcome = await finishing;
		expect(outcome.accepted.map(input => input.id)).toEqual(["steer-1"]);
		expect(channel.accepted).toEqual([steer]);
		expect(channel.deferred).toEqual([]);
		expect(queue).toEqual([]);
	});
	it("records accepted steering right after the steered response and holds later input for the next boundary", async () => {
		let claimedView: string[] | undefined;
		const { contexts, transcript } = await runSteered(async (live, queue) => {
			queue.push(createUserMessage("use tabs"));
			const claim = await live.claim(new AbortController().signal);
			claimedView = claim?.messages.map(textOf);
			claim?.accept();
			// Queued after the server took the first steer.
			queue.push(createUserMessage("also lint"));
		});

		// The provider submits the same bytes the transcript later replays.
		expect(claimedView).toEqual(["<user>use tabs</user>"]);
		// The continuation carries only the delivered steering…
		expect(contexts[1]).toEqual(["<user>start</user>", "first", "<user>use tabs</user>"]);
		// …and later input arrives at the following boundary.
		expect(contexts[2]).toEqual([
			"<user>start</user>",
			"first",
			"<user>use tabs</user>",
			"second",
			"<user>also lint</user>",
		]);
		// Only the delivered steer is marked in the transcript.
		expect(liveSteeredTexts(transcript)).toEqual(["use tabs"]);
	});

	it("delivers rejected steering at the next boundary ahead of input queued after it", async () => {
		const { contexts, transcript } = await runSteered(async (live, queue) => {
			queue.push(createUserMessage("use tabs"));
			const claim = await live.claim(new AbortController().signal);
			queue.push(createUserMessage("also lint"));
			claim?.reject();
		});

		expect(contexts[1]).toEqual(["<user>start</user>", "first", "<user>use tabs</user>", "<user>also lint</user>"]);
		expect(contexts).toHaveLength(2);
		expect(liveSteeredTexts(transcript)).toEqual([]);
	});

	it("keeps steering whose provider view is not a user message out of the live response", async () => {
		let claimed = true;
		const card: AgentMessage = {
			role: "custom",
			customType: "note",
			content: "heads up",
			display: true,
			timestamp: Date.now(),
		};
		const { contexts } = await runSteered(
			async live => {
				claimed = (await live.claim(new AbortController().signal)) !== undefined;
			},
			{ initialQueue: [card] },
		);

		expect(claimed).toBe(false);
		expect(contexts[1]).toEqual(["<user>start</user>", "first", "heads up"]);
	});

	it("soft-interrupts the tool batch of the response that took live steering", async () => {
		// The steer left the queue, so the batch's queue peek never sees it; the
		// loop must still signal foreground tools (bash auto-backgrounds on it).
		const schema = type({});
		let steeringSignal: AbortSignal | undefined;
		let softInterrupted = false;
		const work: AgentTool<typeof schema, Record<string, never>> = {
			name: "work",
			label: "Work",
			description: "Foreground work that yields to steering",
			parameters: schema,
			async execute() {
				// The steer is pending before the batch starts, so the signal is
				// already up when the tool runs.
				softInterrupted = steeringSignal?.aborted === true;
				return { content: [{ type: "text", text: "worked" }], details: {} };
			},
		};
		const { contexts, transcript } = await runSteered(
			async (live, queue) => {
				queue.push(createUserMessage("use tabs"));
				(await live.claim(new AbortController().signal))?.accept();
			},
			{
				first: { content: [{ type: "toolCall", id: "call-1", name: "work", arguments: {} }] },
				tools: [work],
				getToolContext: toolCall => {
					steeringSignal = toolCall?.steeringSignal;
					return { toolCall } as AgentToolContext;
				},
			},
		);

		expect(softInterrupted).toBe(true);
		expect(contexts[1]).toEqual(["<user>start</user>", "", "", "<user>use tabs</user>"]);
		expect(liveSteeredTexts(transcript)).toEqual(["use tabs"]);
	});

	it("requeues live steering when its run aborts, even after the queue was replaced", async () => {
		// The steer left the queue for the response (before any ack); queue edits
		// (Alt+Up, Esc's clear) must not lose it, and the empty-submit interrupt redelivers it.
		const { agent, steer, running } = await startLiveSteeredRun();
		expect(agent.peekSteeringQueue()).toEqual([]);
		expect(agent.peekLiveSteeredMessages()).toEqual([steer]);

		agent.replaceQueues([], []);
		agent.abort();
		await running;

		expect(agent.peekLiveSteeredMessages()).toEqual([]);
		expect(agent.peekSteeringQueue()).toEqual([steer]);
	});

	it("snapshots a converting claim on abort and delivers it once on continuation", async () => {
		const converting = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const mock = createMockModel({ responses: [{ content: ["continued"] }] });
		let claim: Promise<void> | undefined;
		let channel: LiveSteeringChannel | undefined;
		let calls = 0;
		const contexts: string[][] = [];
		const agent = new Agent({
			initialState: { model: mock.model },
			convertToLlm: async messages => {
				if (messages.length === 1 && userText(messages[0]) === "use tabs") {
					converting.resolve();
					await release.promise;
				}
				return convertToLlm(messages);
			},
			streamFn: async (model, context, options) => {
				contexts.push(context.messages.map(textOf));
				if (++calls !== 1) return mock.stream(model, context, options);
				const live = options?.liveSteering;
				const signal = options?.signal;
				if (!(live instanceof LiveSteeringChannel) || !signal) throw new Error("missing steering channel");
				channel = live;
				const stream = new AssistantMessageEventStream();
				// The provider cancellation completes without waiting for conversion.
				signal.addEventListener("abort", () => stream.fail(new Error("aborted")), { once: true });
				claim = live.wait(signal).then(async () => {
					(await live.claim(signal))?.reject();
				});
				return stream;
			},
		});
		const steer = createUserMessage("use tabs");
		const running = agent.prompt("start");
		agent.steer(steer);
		await converting.promise;
		try {
			agent.abort();
			await running;
			// Read at the same terminal boundary as agent-loop's deferred snapshot,
			// before the abandoned conversion is allowed to finish.
			const snapshot = [...(channel?.deferred ?? [])];
			expect(snapshot).toEqual([steer]);
			expect(agent.peekSteeringQueue()).toEqual([steer]);
			await claim;
			release.resolve();
			await agent.continue();
			await claim;
			expect(contexts[1].filter(text => text === "<user>use tabs</user>")).toHaveLength(1);
			expect(channel?.deferred).toEqual(snapshot);
			expect(channel?.accepted).toEqual([]);
			expect(agent.peekSteeringQueue()).toEqual([]);
		} finally {
			release.resolve();
			await claim;
		}
	});

	it("drops withdrawn live steering from the aborted run", async () => {
		// Esc hands the steer back to the editor: the abort must neither requeue nor record it.
		const { agent, steer, running } = await startLiveSteeredRun();

		expect(agent.withdrawUndeliveredQueuedMessages()).toEqual({ steering: [steer], followUp: [] });
		agent.abort();
		await running;

		expect(agent.peekSteeringQueue()).toEqual([]);
		expect(agent.peekUndeliveredQueuedMessages()).toEqual([]);
		expect(agent.state.messages).not.toContain(steer);
	});

	it("drops the prepared context of withdrawn live steering carried into the next turn", async () => {
		// The steer was taken live by a response that ended normally, so the next turn carries it
		// with its prepared context; Esc before that turn's model call must withdraw both.
		const mock = createMockModel({ responses: [{ content: ["first"] }, { content: ["second"] }] });
		let calls = 0;
		const agent = new Agent({
			initialState: { model: mock.model },
			streamFn: async (model, context, options) => {
				if (++calls === 1) {
					const live = options?.liveSteering;
					const signal = options?.signal;
					if (!live || !signal) throw new Error("live steering was not offered");
					await live.wait(signal);
					(await live.claim(signal))?.accept();
				}
				return mock.stream(model, context, options);
			},
		});
		agent.prepareQueuedMessages = messages => ({
			commit: () => messages.map(message => createUserMessage(`context for ${userText(message)}`)),
		});
		const held = Promise.withResolvers<void>();
		let beforeCalls = 0;
		agent.addBeforeModelCall(async (_context, signal) => {
			if (++beforeCalls !== 2) return;
			held.resolve();
			await new Promise<void>(resolve => signal?.addEventListener("abort", () => resolve(), { once: true }));
		});
		const ended: AgentMessage[][] = [];
		const started: AgentMessage[] = [];
		agent.subscribe(event => {
			if (event.type === "agent_end") ended.push(event.messages);
			if (event.type === "message_start") started.push(event.message);
		});
		const running = agent.prompt("start");
		const steer = createUserMessage("use tabs");
		agent.steer(steer);
		await held.promise;

		expect(agent.withdrawUndeliveredQueuedMessages()).toEqual({ steering: [steer], followUp: [] });
		agent.abort();
		await running;

		const userTexts = (messages: AgentMessage[]) => messages.filter(m => m.role === "user").map(userText);
		expect(userTexts(agent.state.messages)).toEqual(["start"]);
		expect(userTexts(started)).toEqual(["start"]);
		expect(ended.flatMap(userTexts)).toEqual(["start"]);
		expect(agent.peekSteeringQueue()).toEqual([]);
	});
});

function frame(data: Uint8Array): Buffer {
	const result = Buffer.alloc(5 + data.length);
	result.writeUInt32BE(data.length, 1);
	result.set(data, 5);
	return result;
}

it.each(["open", "completed"])(
	"pairs %s hosted calls on cancellation in the agent transcript and continuation",
	async mode => {
		const callIds = mode === "open" ? ["open-hosted"] : ["A", "B"];
		const converting = Promise.withResolvers<void>();
		const releaseConversion = Promise.withResolvers<void>();
		const sink: string[] = [];
		const sessions = new Set<http2.ServerHttp2Session>();
		const server = http2.createServer();
		server.on("session", session => {
			sessions.add(session);
			session.on("close", () => sessions.delete(session));
		});
		server.on("stream", (socket: http2.ServerHttp2Stream) => {
			socket.respond({ ":status": 200, "content-type": "application/connect+proto" });
			socket.once("data", () => {
				const frames: Buffer[] = [];
				for (const callId of callIds) {
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
					if (mode === "completed")
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
				socket.write(Buffer.concat(frames));
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
			convertToLlm: async messages => {
				if (messages.length === 1 && userText(messages[0]) === "use tabs") {
					converting.resolve();
					await releaseConversion.promise;
				}
				return convertToLlm(messages);
			},
			streamFn: (_model, context, options) => {
				if (++calls > 1) {
					continuation = context.messages;
					return mock.stream(mock.model, context, options);
				}
				return streamCursor(model, context, {
					...options,
					onToolResult: toolResult => {
						sink.push(toolResult.toolCallId);
						return options?.cursorOnToolResult?.(toolResult) ?? toolResult;
					},
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
			if (mode === "completed") {
				agent.steer(createUserMessage("use tabs"));
				await converting.promise;
			} else await opened.promise;
			agent.abort();
			await running;
			const paired = agent.state.messages.filter(message => message.role === "toolResult");
			expect(paired.map(message => message.toolCallId)).toEqual(callIds);
			if (mode === "open") expect(paired[0]?.isError).toBe(true);
			await agent.continue();
			expect(
				continuation.flatMap(message =>
					message.role === "assistant"
						? message.content.flatMap(block => (block.type === "toolCall" ? [block.id] : []))
						: [],
				),
			).toEqual(callIds);
			expect(continuation.flatMap(message => (message.role === "toolResult" ? [message.toolCallId] : []))).toEqual(
				callIds,
			);
			const snapshot = [...agent.state.messages];
			releaseConversion.resolve();
			const settled = Promise.withResolvers<void>();
			setImmediate(settled.resolve);
			await settled.promise;
			expect(sink).toEqual(callIds);
			expect(agent.state.messages).toEqual(snapshot);
		} finally {
			releaseConversion.resolve();
			agent.abort();
			await running;
			for (const session of sessions) session.destroy();
			server.close();
		}
	},
);
