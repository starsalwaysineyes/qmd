export type ApiRole = "embed" | "rerank";

export type ApiConfig = {
  baseUrl: string;
  apiKey: string;
  timeoutMs: number;
};

export type ApiEmbeddingResult = {
  embedding: number[];
  model: string;
};

export type ApiRerankInputDocument = {
  file: string;
  text: string;
};

export type ApiRerankResult = {
  file: string;
  score: number;
  index: number;
};

const DEFAULT_API_BASE_URL = "https://api.siliconflow.com/v1";
const DEFAULT_API_TIMEOUT_MS = 60_000;

function roleEnvPrefix(role: ApiRole): "QMD_EMBED" | "QMD_RERANK" {
  return role === "embed" ? "QMD_EMBED" : "QMD_RERANK";
}

function envFirst(...names: string[]): string | undefined {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  return undefined;
}

function parseTimeout(value: string | undefined): number {
  if (!value) return DEFAULT_API_TIMEOUT_MS;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) return DEFAULT_API_TIMEOUT_MS;
  return parsed;
}

export function isLikelyLocalModel(model: string): boolean {
  const normalized = model.trim();
  if (!normalized) return false;
  if (normalized.startsWith("hf:")) return true;
  if (normalized.endsWith(".gguf")) return true;
  if (normalized.startsWith("/") || normalized.startsWith("./") || normalized.startsWith("../") || normalized.startsWith("~")) return true;
  return false;
}

export function isDisabledModel(model: string): boolean {
  return ["", "none", "off", "false", "disabled"].includes(model.trim().toLowerCase());
}

function looksLikeRemoteModelName(model: string): boolean {
  const normalized = model.trim();
  if (!normalized || isLikelyLocalModel(normalized) || isDisabledModel(normalized)) return false;
  return normalized.includes("/");
}

export function shouldUseApiProvider(role: ApiRole, model: string): boolean {
  const prefix = roleEnvPrefix(role);
  const provider = envFirst(`${prefix}_PROVIDER`, "QMD_API_PROVIDER")?.toLowerCase();
  if (provider) {
    if (["local", "llama", "llama.cpp", "node-llama-cpp"].includes(provider)) return false;
    if (["api", "openai", "siliconflow", "remote", "http"].includes(provider)) return true;
  }

  if (isLikelyLocalModel(model)) return false;
  return looksLikeRemoteModelName(model);
}

export function hasApiProviderConfig(role: ApiRole): boolean {
  const prefix = roleEnvPrefix(role);
  const provider = envFirst(`${prefix}_PROVIDER`, "QMD_API_PROVIDER")?.toLowerCase();
  if (provider && ["api", "openai", "siliconflow", "remote", "http"].includes(provider)) return true;
  return !!envFirst(`${prefix}_API_KEY`, "QMD_API_KEY", "SILICONFLOW_API_KEY", `${prefix}_API_BASE`, "QMD_API_BASE");
}

export function resolveApiConfig(role: ApiRole): ApiConfig {
  const prefix = roleEnvPrefix(role);
  const apiKey = envFirst(`${prefix}_API_KEY`, "QMD_API_KEY", "SILICONFLOW_API_KEY");
  if (!apiKey) {
    throw new Error(
      `${prefix}_API_KEY or QMD_API_KEY is required for API ${role === "embed" ? "embeddings" : "reranking"}`
    );
  }

  const baseUrl = envFirst(`${prefix}_API_BASE`, "QMD_API_BASE") ?? DEFAULT_API_BASE_URL;
  return {
    baseUrl: baseUrl.replace(/\/+$/, ""),
    apiKey,
    timeoutMs: parseTimeout(envFirst(`${prefix}_API_TIMEOUT_MS`, "QMD_API_TIMEOUT_MS")),
  };
}

function endpointUrl(config: ApiConfig, path: string): string {
  return `${config.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

async function postJson(config: ApiConfig, path: string, body: unknown): Promise<unknown> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  timeout.unref();

  try {
    const response = await fetch(endpointUrl(config, path), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const text = await response.text();
    let payload: unknown;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { error: text };
    }

    if (!response.ok) {
      const details = typeof payload === "object" && payload !== null
        ? JSON.stringify(payload).slice(0, 500)
        : String(payload).slice(0, 500);
      throw new Error(`API request failed (${response.status}): ${details}`);
    }

    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

function parseEmbeddingResponse(payload: unknown): number[][] {
  const record = asRecord(payload);
  const data = Array.isArray(record.data) ? record.data : [];
  const rows = data
    .map((item, fallbackIndex) => {
      const row = asRecord(item);
      return {
        index: typeof row.index === "number" ? row.index : fallbackIndex,
        embedding: Array.isArray(row.embedding) ? row.embedding.map(Number) : null,
      };
    })
    .filter((row): row is { index: number; embedding: number[] } => Array.isArray(row.embedding));

  rows.sort((a, b) => a.index - b.index);
  return rows.map(row => row.embedding);
}

export async function embedWithApi(texts: string[], model: string): Promise<ApiEmbeddingResult[]> {
  if (texts.length === 0) return [];

  const config = resolveApiConfig("embed");
  const dimensions = envFirst("QMD_EMBED_DIMENSIONS");
  const payload: Record<string, unknown> = {
    model,
    input: texts,
  };
  if (dimensions) {
    const parsed = Number.parseInt(dimensions, 10);
    if (Number.isInteger(parsed) && parsed > 0) payload.dimensions = parsed;
  }

  const response = await postJson(config, "/embeddings", payload);
  const embeddings = parseEmbeddingResponse(response);
  if (embeddings.length !== texts.length) {
    throw new Error(`Embedding API returned ${embeddings.length} vectors for ${texts.length} inputs`);
  }
  return embeddings.map(embedding => ({ embedding, model }));
}

function parseRerankResponse(payload: unknown, documents: ApiRerankInputDocument[]): ApiRerankResult[] {
  const record = asRecord(payload);
  const rawResults = Array.isArray(record.results)
    ? record.results
    : Array.isArray(record.data)
      ? record.data
      : Array.isArray(record.result)
        ? record.result
        : [];

  return rawResults.map((item, fallbackIndex) => {
    const row = asRecord(item);
    const indexValue = row.index ?? row.document_index ?? row.documentIndex ?? fallbackIndex;
    const index = typeof indexValue === "number" ? indexValue : Number(indexValue);
    const scoreValue = row.relevance_score ?? row.relevanceScore ?? row.score ?? 0;
    const score = typeof scoreValue === "number" ? scoreValue : Number(scoreValue);
    const safeIndex = Number.isInteger(index) && index >= 0 && index < documents.length ? index : fallbackIndex;
    return {
      file: documents[safeIndex]?.file ?? documents[fallbackIndex]?.file ?? "",
      score: Number.isFinite(score) ? score : 0,
      index: safeIndex,
    };
  }).filter(result => result.file);
}

export async function rerankWithApi(
  query: string,
  documents: ApiRerankInputDocument[],
  model: string
): Promise<ApiRerankResult[]> {
  if (documents.length === 0) return [];

  const config = resolveApiConfig("rerank");
  const response = await postJson(config, "/rerank", {
    model,
    query,
    documents: documents.map(document => document.text),
    return_documents: false,
    top_n: documents.length,
  });

  const results = parseRerankResponse(response, documents);
  if (results.length === 0) {
    throw new Error("Rerank API returned no results");
  }
  return results.sort((a, b) => b.score - a.score);
}
