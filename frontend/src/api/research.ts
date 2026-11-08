export type RetrievedChunk = {
  score?: number;
  paper_id?: string;
  title?: string;
  section_title?: string;
  chunk_text?: string;
  chunk_index?: number;
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

export async function runResearch(query: string): Promise<ResearchResponse> {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      focus_areas: [],
    }),
  });

  if (!response.ok) {
    throw new Error(`Research request failed with status ${response.status}`);
  }

  return response.json();
}
