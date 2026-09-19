import { createContext } from "react";

/** Marks the canvas puts on nodes without owning their data.
 *
 * Entry and error are properties of the *graph*, not of a node, and they
 * change far more often than the nodes do: every compile, every edge added.
 * Carrying them in `node.data` would mean rewriting every node object to move
 * a highlight, so they travel by context instead and the node list stays what
 * it says it is, the list of steps.
 */
export interface Marks {
  /** The node the conversation starts on, as the compiler would infer it. */
  entry: string | null;
  /** The node the last compile error named, if it named one. */
  problem: string | null;
}

export const FlowMarks = createContext<Marks>({ entry: null, problem: null });
