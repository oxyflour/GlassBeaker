'use client'

import type { ChangeEvent } from "react";

import type { RobotModelKey } from "./robot-model";

export function RobotModelSelect({
  activeRobotModelKey,
  onChange,
}: {
  activeRobotModelKey: RobotModelKey | null
  onChange: (key: RobotModelKey) => void
}) {
  function handleChange(event: ChangeEvent<HTMLSelectElement>) {
    onChange(event.target.value as RobotModelKey);
  }

  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <label className="mr-2 text-sm" htmlFor="robot-model">Robot model</label>
    <select
      id="robot-model"
      className="rounded border border-white/20 bg-black/40 px-2 py-1 text-sm"
      onChange={ handleChange }
      value={ activeRobotModelKey ?? "" }>
      <option disabled value="">Unknown URL model</option>
      <option value="r1pro">r1pro</option>
      <option value="moz1">moz1</option>
    </select>
  </div>;
}
