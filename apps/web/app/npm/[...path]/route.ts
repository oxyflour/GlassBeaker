import { NextRequest } from 'next/server'

export const dynamic = 'force-dynamic'
export const revalidate = 0

const UPSTREAM_BASE = process.env.NPM_MIRROR || 'https://registry.npmmirror.com'

// 生产里最好改成你的前端域名，或者做白名单判断
const ALLOW_ORIGIN = '*'

function isJsonResponse(contentType: string | null, pathname: string) {
    if (!contentType) return pathname.endsWith('.json')

    const ct = contentType.toLowerCase()
    return (
        ct.includes('application/json') ||
        ct.includes('application/vnd.npm.install-v1+json') ||
        ct.includes('+json')
    )
}

function applyCorsHeaders(headers: Headers, req: NextRequest) {
    const origin = req.headers.get('origin')

    // 如果你要带 cookie / auth，就不能用 *，要回显具体 origin
    headers.set(
        'access-control-allow-origin',
        ALLOW_ORIGIN === '*' ? '*' : origin || ALLOW_ORIGIN
    )
    headers.set('access-control-allow-methods', 'GET,HEAD,OPTIONS')
    headers.set(
        'access-control-allow-headers',
        'Content-Type, Authorization, Accept, Origin, User-Agent'
    )
    headers.set('access-control-expose-headers', 'Content-Type, Content-Length')
    headers.set('access-control-max-age', '86400')

    // 如果后面要支持 cookie/认证，再打开这个，同时 allow-origin 不能是 *
    // headers.set('access-control-allow-credentials', 'true')

    // 避免 CDN / cache 把不同 Origin 的响应混掉
    headers.append('vary', 'Origin')
}

function copyResponseHeaders(upstreamHeaders: Headers, req: NextRequest) {
    const headers = new Headers(upstreamHeaders)

    // 这些最容易和新 body 不一致
    headers.delete('content-length')
    headers.delete('content-encoding')
    headers.delete('transfer-encoding')

    headers.set('cache-control', 'no-store')
    applyCorsHeaders(headers, req)
    return headers
}

async function proxy(req: NextRequest, path: string[]) {
    const upstreamUrl = new URL(`${UPSTREAM_BASE}/${path.join('/')}`)
    const upstreamRes = await fetch(upstreamUrl, {
        method: req.method,
        cache: 'no-store',
        redirect: 'follow',
    })

    const contentType = upstreamRes.headers.get('content-type')
    if (!isJsonResponse(contentType, upstreamUrl.pathname)) {
        return new Response(upstreamRes.body, {
            status: upstreamRes.status,
            statusText: upstreamRes.statusText,
            headers: copyResponseHeaders(upstreamRes.headers, req),
        })
    }

    const json = await upstreamRes.json(),
        url = req.nextUrl
    for (const value of Object.values(json.versions || { })) {
        const item = value as { dist?: { tarball?: string } }
        if (item?.dist?.tarball) {
            const ret = new URL(item?.dist?.tarball)
            item.dist.tarball = `${url.origin}/npm${ret.pathname}`
        }
    }

    return new Response(JSON.stringify(json), {
        status: upstreamRes.status,
        statusText: upstreamRes.statusText,
        headers: copyResponseHeaders(upstreamRes.headers, req),
    })
}

export async function GET(
    req: NextRequest,
    context: { params: Promise<{ path: string[] }> }
) {
    const { path } = await context.params
    return proxy(req, path)
}

export async function HEAD(
    req: NextRequest,
    context: { params: Promise<{ path: string[] }> }
) {
    const { path } = await context.params
    return proxy(req, path)
}

export async function OPTIONS(req: NextRequest) {
    const headers = new Headers()
    applyCorsHeaders(headers, req)
    return new Response(null, {
        status: 204,
        headers,
    })
}
