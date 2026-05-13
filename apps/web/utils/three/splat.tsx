"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react"
import { useThree } from "@react-three/fiber"
import { Group } from "three"
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark"

export const SparkRendererBridge = forwardRef<SparkRenderer>(function SparkRendererBridge(_, ref) {
    const { gl, invalidate } = useThree()
    const [spark] = useState(() => new SparkRenderer({
        renderer: gl,
        onDirty: invalidate,
    }))

    useImperativeHandle(ref, () => spark, [spark])

    useEffect(() => {
        spark.renderer = gl
        spark.onDirty = invalidate
        spark.renderOrder = 1000
        return () => {
            spark.dispose()
        }
    }, [gl, invalidate, spark])

    return <primitive object={ spark } dispose={ null } />
})

export function SparkSplat({ onError, onReadyChange, url }: {
    onError?: (error: unknown) => void
    onReadyChange?: (ready: boolean) => void
    url: string
}) {
    const ref = useRef<Group>(null)

    useEffect(() => {
        if (!ref.current) {
                return () => { }
        }
        const root = ref.current
        return attachSparkSplat({
            createSplat: (options) => new SplatMesh(options),
            onError,
            onReadyChange,
            root: {
                add: (object) => {
                    root.add(object)
                },
                remove: (object) => {
                    root.remove(object)
                },
            },
            url,
        })
    }, [onError, onReadyChange, url])

    return <group ref={ ref } />
}

type SparkSplatLike = {
    initialized: Promise<unknown>
    dispose?: () => void
}

export function attachSparkSplat<TSplat extends SparkSplatLike>({
    createSplat,
    onError,
    onReadyChange,
    root,
    url,
}: {
    createSplat: (options: { url: string }) => TSplat
    onError?: (error: unknown) => void
    onReadyChange?: (ready: boolean) => void
    root: {
        add: (object: TSplat) => void
        remove: (object: TSplat) => void
    }
    url: string
}) {
    let active = true
    onReadyChange?.(false)
    const splat = createSplat({ url })
    root.add(splat)
    void splat.initialized
        .then(() => {
            if (active) {
                onReadyChange?.(true)
            }
        })
        .catch((error) => {
            if (active) {
                onError?.(error)
            }
        })
    return () => {
        active = false
        root.remove(splat)
        splat.dispose?.()
        onReadyChange?.(false)
    }
}
