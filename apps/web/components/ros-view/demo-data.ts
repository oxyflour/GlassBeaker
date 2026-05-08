export type RosTopic = RosImageTopic | RosPlotTopic | RosStateTopic;

export interface RosImageTopic {
  id: string;
  kind: "image";
  label: string;
  description: string;
  src: string;
}

export interface RosPlotTopic {
  id: string;
  kind: "plot";
  label: string;
  description: string;
  unit: string;
  timestamps: string[];
  series: Array<{ label: string; color: string; values: number[] }>;
}

export interface RosStateTopic {
  id: string;
  kind: "state";
  label: string;
  description: string;
  fields: Array<{ label: string; value: string; tone?: "good" | "warn" | "alert" }>;
}

export const EMPTY_TOPIC: RosStateTopic = {
  id: "__empty__",
  kind: "state",
  label: "No ROS topics",
  description: "ROS bridge is disconnected or no topics have been discovered yet.",
  fields: [
    { label: "Status", value: "Waiting for /python/ros_view/state", tone: "warn" },
    { label: "Topics", value: "0 discovered", tone: "warn" },
  ],
};

export function getTopicById(topicId: string, topics: RosTopic[] = []): RosTopic {
  return topics.find((topic) => topic.id === topicId) ?? topics[0] ?? EMPTY_TOPIC;
}

export function getNextTopicId(topicId: string, topics: RosTopic[] = []): string {
  if (topics.length === 0) return EMPTY_TOPIC.id;
  const index = topics.findIndex((topic) => topic.id === topicId);
  return topics[(index + 1 + topics.length) % topics.length]?.id ?? topics[0]?.id ?? EMPTY_TOPIC.id;
}
