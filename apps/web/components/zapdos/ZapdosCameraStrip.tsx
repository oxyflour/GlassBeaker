'use client'

import { useEffect, useRef, useState } from "react";

import { getZapdosRuntimeErrorMessage } from "./zapdos-runtime";
import {
  buildCompositeCameraSlices,
  buildCompositeCameraStreamUrl,
  loadCameraNames,
} from "./zapdos-camera-preview";

export function ZapdosCameraStrip({
  sess,
  onRuntimeError,
}: {
  sess: string;
  onRuntimeError: (message: string) => void;
}) {
  const [cameras, setCameras] = useState<string[]>([]);
  const [reportedStreamError, setReportedStreamError] = useState(false);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});

  useEffect(() => {
    let cancelled = false;
    void loadCameraNames(sess)
      .then(next => {
        if (cancelled) return;
        setCameras(next);
        setReportedStreamError(false);
      })
      .catch(error => {
        if (!cancelled) onRuntimeError(getZapdosRuntimeErrorMessage(error));
      });
    return () => {
      cancelled = true;
    };
  }, [onRuntimeError, sess]);

  useEffect(() => {
    if (cameras.length === 0) return;
    let frameId = 0;
    const draw = () => {
      const image = imageRef.current;
      if (image && image.naturalWidth > 0 && image.naturalHeight > 0) {
        for (const slice of buildCompositeCameraSlices(cameras, image.naturalWidth, image.naturalHeight)) {
          const canvas = canvasRefs.current[slice.camera];
          const context = canvas?.getContext("2d");
          if (!canvas || !context) continue;
          if (canvas.width !== slice.width || canvas.height !== slice.height) {
            canvas.width = slice.width;
            canvas.height = slice.height;
          }
          context.drawImage(
            image,
            slice.left,
            0,
            slice.width,
            slice.height,
            0,
            0,
            canvas.width,
            canvas.height,
          );
        }
      }
      frameId = window.requestAnimationFrame(draw);
    };
    frameId = window.requestAnimationFrame(draw);
    return () => window.cancelAnimationFrame(frameId);
  }, [cameras]);

  if (cameras.length === 0) return null;
  return <div className="pointer-events-none absolute bottom-0 left-0 w-full p-4">
    <img
      alt=""
      aria-hidden="true"
      className="pointer-events-none absolute h-px w-px opacity-0"
      onError={ () => {
        if (reportedStreamError) return;
        setReportedStreamError(true);
        onRuntimeError("Camera stream failed");
      } }
      onLoad={ () => setReportedStreamError(false) }
      ref={ imageRef }
      src={ buildCompositeCameraStreamUrl(sess) } />
    <div className="flex justify-center gap-3">
      {cameras.map(camera => (
        <div className="overflow-hidden rounded-lg border border-white/20 bg-black/50 shadow-lg" key={ camera }>
          <div className="border-b border-white/10 px-2 py-1 text-xs text-white/70">{camera}</div>
          <canvas
            className="block h-24 w-40 bg-black object-cover"
            ref={ element => {
              canvasRefs.current[camera] = element;
            } } />
        </div>
      ))}
    </div>
  </div>;
}
