'use client'

import { useMemo } from "react"

export function useLocalUUID(key: string) {
    return useMemo(() => {
        // prevent nextjs complaining about localStorage being unavailable during server-side rendering
        if (typeof window === 'undefined') {
            return ''
        }

        let uuid = localStorage.getItem(key)
        if (!uuid) {
            uuid = crypto.randomUUID()
            localStorage.setItem(key, uuid)
        }
        return uuid
    }, [key])
}
