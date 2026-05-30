import { z } from "zod";

export const listManipulationObjectsToolArgsSchema = z.object({}).strict();

export const pickObjectToolArgsSchema = z.object({
  target_query: z.string().trim().min(1).describe("Natural-language description of the object to pick."),
  support_query: z.string().trim().min(1).optional().describe("Optional natural-language description of the support surface holding the target."),
  arm: z.enum(["left", "right"]).default("left").describe("Robot arm to use for the pick."),
}).strict();

export type PickObjectToolArgs = z.infer<typeof pickObjectToolArgsSchema>;
