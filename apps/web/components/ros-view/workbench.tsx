"use client";

import { useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";

import { EMPTY_TOPIC, getNextTopicId, getTopicById, type RosTopic } from "./demo-data";
import { countLeaves, createInitialLayout, removeLeaf, splitLeaf, type PanelNode, updateLeafTopic } from "./layout";
import RosPanelView from "./panel-view";
import { useRosViewState } from "./use-ros-view-state";

export default function RosViewWorkbench() {
  const { topics } = useRosViewState();
  const [layout, setLayout] = useState<PanelNode>(() => createInitialLayout(topics[0]?.id ?? EMPTY_TOPIC.id));
  const canClose = countLeaves(layout) > 1;

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#02060b] text-slate-100">
      <TreeNode
        node={layout}
        topics={topics}
        canClose={canClose}
        onSplit={(panelId, direction, topicId) => setLayout((current) => splitLeaf(current, panelId, direction, topicId))}
        onClose={(panelId) => setLayout((current) => removeLeaf(current, panelId))}
        onTopicChange={(panelId, topicId) => setLayout((current) => updateLeafTopic(current, panelId, topicId))}
      />
    </div>
  );
}

function TreeNode({
  node,
  topics,
  canClose,
  onSplit,
  onClose,
  onTopicChange,
}: {
  node: PanelNode;
  topics: RosTopic[];
  canClose: boolean;
  onSplit: (panelId: string, direction: "right" | "down", topicId: string) => void;
  onClose: (panelId: string) => void;
  onTopicChange: (panelId: string, topicId: string) => void;
}) {
  if (node.kind === "leaf") {
    const topic = getTopicById(node.topicId, topics);
    const nextTopicId = getNextTopicId(node.topicId, topics);
    const options = topics.length > 0 ? topics : [topic];
    return (
      <RosPanelView
        topic={topic}
        topics={options}
        canClose={canClose}
        onTopicChange={(topicId) => onTopicChange(node.id, topicId)}
        onSplitRight={() => onSplit(node.id, "right", nextTopicId)}
        onSplitDown={() => onSplit(node.id, "down", nextTopicId)}
        onClose={() => onClose(node.id)}
      />
    );
  }
  return (
    <Group orientation={node.orientation} style={{ height: "100%", width: "100%" }}>
      <Panel defaultSize={50} minSize={12}>
        <TreeNode node={node.first} topics={topics} canClose={canClose} onSplit={onSplit} onClose={onClose} onTopicChange={onTopicChange} />
      </Panel>
      <Separator style={node.orientation === "horizontal" ? { width: 1, background: "rgba(255,255,255,0.12)" } : { height: 1, background: "rgba(255,255,255,0.12)" }} />
      <Panel defaultSize={50} minSize={12}>
        <TreeNode node={node.second} topics={topics} canClose={canClose} onSplit={onSplit} onClose={onClose} onTopicChange={onTopicChange} />
      </Panel>
    </Group>
  );
}
