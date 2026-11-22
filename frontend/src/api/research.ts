import { fbAuth } from "../firebase";
export type RetrievedChunk = {
  score?: number;
  paper_id?: string;
  title?: string;
  section_title?: string;
  chunk_text?: string;
  chunk_index?: number;
  authors?: string[];
  publish_date?: string | null;
  categories?: string[];
};

export type Citation = {
  paper_id?: string;
  supporting_quote?: string;
};

export type ResearchResponse = {
  answer?: string;
  confidence?: string;
  top_score?: number;
  retrieved_chunks?: RetrievedChunk[];
  citations?: Citation[];
};

export async function runResearch(
  query: string,
  threadId?: string
): Promise<ResearchResponse> {
  const user = fbAuth.currentUser;
  if (!user) throw new Error("You are signed out. Please sign in again.");
  const token = await user.getIdToken(); // refreshes automatically when expired

  const response = await fetch("/api/research", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      query,
      thread_id: threadId,
      focus_areas: [],
    }),
  });

  if (!response.ok) {
    // Show the backend's message (e.g. "Is Ollama running...") instead of just a status code.
    const detail = await response.json().then((b) => b.detail).catch(() => undefined);
    throw new Error(
      typeof detail === "string" ? detail : `Research request failed with status ${response.status}`,
    );
  }

  return response.json();
}
