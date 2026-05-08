export type SplitDirection = "right" | "down";
export type PanelOrientation = "horizontal" | "vertical";

export interface PanelLeaf {
  kind: "leaf";
  id: string;
  topicId: string;
}

export interface PanelSplit {
  kind: "split";
  id: string;
  orientation: PanelOrientation;
  first: PanelNode;
  second: PanelNode;
}

export type PanelNode = PanelLeaf | PanelSplit;

let panelCount = 0;
let splitCount = 0;

function createLeaf(topicId: string): PanelLeaf {
  panelCount += 1;
  return { kind: "leaf", id: `panel-${panelCount}`, topicId };
}

function createSplit(orientation: PanelOrientation, first: PanelNode, second: PanelNode): PanelSplit {
  splitCount += 1;
  return { kind: "split", id: `split-${splitCount}`, orientation, first, second };
}

function orientationFor(direction: SplitDirection): PanelOrientation {
  return direction === "right" ? "horizontal" : "vertical";
}

export function createInitialLayout(topicId: string): PanelNode {
  panelCount = 0;
  splitCount = 0;
  return createLeaf(topicId || "__empty__");
}

export function countLeaves(node: PanelNode): number {
  return node.kind === "leaf" ? 1 : countLeaves(node.first) + countLeaves(node.second);
}

export function splitLeaf(node: PanelNode, targetId: string, direction: SplitDirection, topicId: string): PanelNode {
  if (node.kind === "leaf") {
    return node.id === targetId ? createSplit(orientationFor(direction), node, createLeaf(topicId)) : node;
  }
  return {
    ...node,
    first: splitLeaf(node.first, targetId, direction, topicId),
    second: splitLeaf(node.second, targetId, direction, topicId),
  };
}

export function updateLeafTopic(node: PanelNode, targetId: string, topicId: string): PanelNode {
  if (node.kind === "leaf") {
    return node.id === targetId ? { ...node, topicId } : node;
  }
  return {
    ...node,
    first: updateLeafTopic(node.first, targetId, topicId),
    second: updateLeafTopic(node.second, targetId, topicId),
  };
}

function removeLeafOrNull(node: PanelNode, targetId: string): PanelNode | null {
  if (node.kind === "leaf") {
    return node.id === targetId ? null : node;
  }
  const first = removeLeafOrNull(node.first, targetId);
  const second = removeLeafOrNull(node.second, targetId);
  if (!first && !second) return null;
  if (!first) return second;
  if (!second) return first;
  return { ...node, first, second };
}

export function removeLeaf(node: PanelNode, targetId: string): PanelNode {
  return removeLeafOrNull(node, targetId) ?? node;
}
