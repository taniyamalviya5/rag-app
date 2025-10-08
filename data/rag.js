// rag.js
import fs from "fs/promises";
import { pipeline } from "@xenova/transformers";

// Singleton embedding model loader
let embedder = null;
const getEmbedder = async () => {
  if (!embedder) {
    console.log("🔸 Loading embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("✅ Model loaded");
  }
  return embedder;
};

// Load docs
export const loadDocs = async () => {
  const raw = await fs.readFile("./data/doc.json", "utf8");
  return JSON.parse(raw);
};

// Embed text into vector
export const embedText = async (text) => {
  const model = await getEmbedder();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // Float32Array → normal array
};

// Compute cosine similarity between two vectors
export const cosineSimilarity = (a, b) => {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

// Build embeddings index
export const buildIndex = async () => {
  const docs = await loadDocs();
  const index = [];
  for (const doc of docs) {
    const embedding = await embedText(doc.text);
    index.push({ ...doc, embedding });
  }
  return index;
};

// Search top-k relevant docs
export const search = async (query, index, k = 3) => {
  const qVec = await embedText(query);
  const scored = index.map((doc) => ({
    ...doc,
    score: cosineSimilarity(qVec, doc.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
};
