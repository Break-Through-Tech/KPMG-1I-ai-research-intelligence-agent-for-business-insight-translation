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

export async function research(
  query: string,
  focusAreas: string[],
): Promise<ResearchResponse> {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, focus_areas: focusAreas }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "The research request could not be completed.");
  }

  return response.json() as Promise<ResearchResponse>;
}
