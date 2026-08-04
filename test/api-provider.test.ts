import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { hasApiProviderConfig, resolveApiConfig } from "../src/api-provider.js";

describe("API provider key files", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.unstubAllEnvs();
    for (const directory of tempDirs.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("loads an embedding API key from QMD_EMBED_API_KEY_FILE", () => {
    const directory = mkdtempSync(join(tmpdir(), "qmd-api-key-"));
    tempDirs.push(directory);
    const keyPath = join(directory, "key");
    writeFileSync(keyPath, "file-key\n", { mode: 0o600 });
    vi.stubEnv("QMD_EMBED_API_KEY_FILE", keyPath);
    vi.stubEnv("SILICONFLOW_API_KEY", "fallback-key");
    vi.stubEnv("QMD_EMBED_API_BASE", "http://127.0.0.1:8317/v1/");

    expect(hasApiProviderConfig("embed")).toBe(true);
    expect(resolveApiConfig("embed")).toEqual({
      baseUrl: "http://127.0.0.1:8317/v1",
      apiKey: "file-key",
      timeoutMs: 60_000,
    });
  });

  it("prefers an explicit role API key over the key file", () => {
    const directory = mkdtempSync(join(tmpdir(), "qmd-api-key-"));
    tempDirs.push(directory);
    const keyPath = join(directory, "key");
    writeFileSync(keyPath, "file-key", { mode: 0o600 });
    vi.stubEnv("QMD_EMBED_API_KEY_FILE", keyPath);
    vi.stubEnv("QMD_EMBED_API_KEY", "explicit-key");

    expect(resolveApiConfig("embed").apiKey).toBe("explicit-key");
  });
});
