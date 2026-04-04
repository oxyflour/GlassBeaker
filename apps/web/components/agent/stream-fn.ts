import type { StreamFn } from "@mariozechner/pi-agent-core"

type PiUsage = {
    input: number
    output: number
    cacheRead: number
    cacheWrite: number
    totalTokens: number
    cost: {
        input: number
        output: number
        cacheRead: number
        cacheWrite: number
        total: number
    }
}

type PiStreamEvent =
    | { type: 'start' }
    | { type: 'text_start', contentIndex: number }
    | { type: 'text_delta', contentIndex: number, delta: string }
    | { type: 'text_end', contentIndex: number }
    | { type: 'done', reason: 'stop' | 'length' | 'toolUse', usage: PiUsage }
    | { type: 'error', reason: 'aborted' | 'error', errorMessage?: string, usage: PiUsage }

function createUsage(): PiUsage {
    return {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            total: 0,
        },
    }
}

function createPiStream(model: { api: string, provider: string, id: string }) {
    const partial: any = {
        role: 'assistant' as const,
        stopReason: 'stop',
        content: [] as Array<{ type: 'text', text: string }>,
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: createUsage(),
        timestamp: Date.now(),
    }

    let done = false
    let pending: any[] = []
    let nextResolve: null | ((value: IteratorResult<any>) => void) = null
    let resolveResult = (_message: any) => { }
    const resultPromise = new Promise<any>(resolve => {
        resolveResult = resolve
    })

    return {
        partial,
        push(event: any) {
            if (done) {
                return
            }

            if (nextResolve) {
                const resolve = nextResolve
                nextResolve = null
                resolve({ value: event, done: false })
                return
            }

            pending.push(event)
        },
        finish(message: any) {
            if (done) {
                return
            }

            done = true
            resolveResult(message)
            if (nextResolve) {
                const resolve = nextResolve
                nextResolve = null
                resolve({ value: undefined, done: true })
            }
        },
        async result() {
            return resultPromise
        },
        [Symbol.asyncIterator]() {
            return {
                next() {
                    if (pending.length > 0) {
                        return Promise.resolve({ value: pending.shift(), done: false })
                    }

                    if (done) {
                        return Promise.resolve({ value: undefined, done: true })
                    }

                    return new Promise<IteratorResult<any>>(resolve => {
                        nextResolve = resolve
                    })
                }
            }
        }
    }
}

function processPiStreamEvent(stream: ReturnType<typeof createPiStream>, event: PiStreamEvent) {
    switch (event.type) {
        case 'start':
            stream.push({ type: 'start', partial: stream.partial })
            return
        case 'text_start':
            stream.partial.content[event.contentIndex] = { type: 'text', text: '' }
            stream.push({ type: 'text_start', contentIndex: event.contentIndex, partial: stream.partial })
            return
        case 'text_delta': {
            const content = stream.partial.content[event.contentIndex]
            if (content) {
                content.text += event.delta
            }
            stream.push({
                type: 'text_delta',
                contentIndex: event.contentIndex,
                delta: event.delta,
                partial: stream.partial
            })
            return
        }
        case 'text_end': {
            const content = stream.partial.content[event.contentIndex]
            stream.push({
                type: 'text_end',
                contentIndex: event.contentIndex,
                content: content?.text || '',
                partial: stream.partial
            })
            return
        }
        case 'done': {
            stream.partial.stopReason = event.reason
            stream.partial.usage = event.usage
            stream.push({ type: 'done', reason: event.reason, message: stream.partial })
            stream.finish(stream.partial)
            return
        }
        case 'error':
            stream.partial.stopReason = event.reason
            stream.partial.errorMessage = event.errorMessage
            stream.partial.usage = event.usage
            stream.push({ type: 'error', reason: event.reason, error: stream.partial })
            stream.finish(stream.partial)
            return
    }
}

async function readErrorMessage(response: Response) {
    try {
        const data = await response.json()
        if (typeof data?.error === 'string' && data.error) {
            return data.error
        }
    } catch {
    }

    return `Pi route error: ${response.status} ${response.statusText}`
}

export const streamFn: StreamFn = async (model, context, options) => {
    const stream = createPiStream(model)

    try {
        const response = await fetch('/api/pi', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model,
                context,
                options: {
                    apiKey: options?.apiKey,
                    reasoning: options?.reasoning,
                },
            }),
            signal: options?.signal,
        })

        if (!response.ok) {
            processPiStreamEvent(stream, {
                type: 'error',
                reason: options?.signal?.aborted ? 'aborted' : 'error',
                errorMessage: await readErrorMessage(response),
                usage: createUsage(),
            })
            return stream as any
        }

        const reader = response.body?.getReader()
        if (!reader) {
            processPiStreamEvent(stream, {
                type: 'error',
                reason: 'error',
                errorMessage: 'Pi route returned an empty stream.',
                usage: createUsage(),
            })
            return stream as any
        }

        const decoder = new TextDecoder()
        let buffer = ''
        let finished = false

        void (async () => {
            try {
                while (true) {
                    const { done, value } = await reader.read()
                    if (done) {
                        break
                    }

                    buffer += decoder.decode(value, { stream: true })
                    const lines = buffer.split('\n')
                    buffer = lines.pop() || ''

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) {
                            continue
                        }

                        const event = JSON.parse(line.slice(6).trim()) as PiStreamEvent
                        processPiStreamEvent(stream, event)
                        if (event.type === 'done' || event.type === 'error') {
                            finished = true
                        }
                    }
                }

                if (!finished) {
                    processPiStreamEvent(stream, {
                        type: 'error',
                        reason: options?.signal?.aborted ? 'aborted' : 'error',
                        errorMessage: 'Pi stream ended unexpectedly.',
                        usage: createUsage(),
                    })
                }
            } catch (error) {
                processPiStreamEvent(stream, {
                    type: 'error',
                    reason: options?.signal?.aborted ? 'aborted' : 'error',
                    errorMessage: error instanceof Error ? error.message : String(error),
                    usage: createUsage(),
                })
            }
        })()

        return stream as any
    } catch (error) {
        processPiStreamEvent(stream, {
            type: 'error',
            reason: options?.signal?.aborted ? 'aborted' : 'error',
            errorMessage: error instanceof Error ? error.message : String(error),
            usage: createUsage(),
        })
        return stream as any
    }
}