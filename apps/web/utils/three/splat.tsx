"use client";

import { useEffect, useRef, useState } from "react"
import { useThree } from "@react-three/fiber"
import { Group } from "three"
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark"

export function SparkRendererBridge() {
    const { gl, invalidate } = useThree()
    const [spark] = useState(() => new SparkRenderer({
        renderer: gl,
        onDirty: invalidate,
    }))

    useEffect(() => {
        spark.renderer = gl
        spark.onDirty = invalidate
        spark.renderOrder = 1000
        return () => {
            spark.dispose()
        }
    }, [gl, invalidate, spark])

    return <primitive object={ spark } dispose={ null } />
}

export function SparkSplat({ url }: {
    url: string
}) {
    const ref = useRef<Group>(null)

    useEffect(() => {
        if (!ref.current) {
                return () => { }
        }

        const splat = new SplatMesh({ url })
        ref.current.add(splat)

        return () => {
            ref.current?.remove(splat)
            splat.dispose?.()
        }
    }, [url])

    return <group ref={ ref } />
}
