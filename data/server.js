// server.js
import express from "express";
import cors from "cors";
import { buildIndex, search } from "./rag.js";

const app = express();
app.use(cors());
app.use(express.json());

let index = null;

// Build index once on startup
const init = async () => {
  index = await buildIndex();
  console.log(`📚 Index built with ${index.length} docs`);
};
await init();

// POST /query — retrieve top docs
app.post("/query", async (req, res) => {
  try {
    const { query, k = 3 } = req.body;
    if (!query) return res.status(400).json({ error: "Query required" });
    const results = await search(query, index, k);
    res.json({ query, results });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(3000, () =>
  console.log("🚀 RAG API running on http://localhost:3000")
);
