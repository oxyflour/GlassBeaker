'use client'

import dynamic from "next/dynamic";

const Pi = dynamic(() => import("../../../components/agent/pi"), { ssr: false });

export default function PiDemoPage() {
  return <Pi />;
}
