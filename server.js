import express from "express";
import * as dotenv from "dotenv";
import { chunks, generateAnswer, getEmbedding, search } from "./ragServer.js";
import { dot } from "@xenova/transformers";

dotenv.config();

const app = express();
app.use(express.json());

app.post("/query", async (req, res) => {
  const { question } = req.body;
  const queryEmbedding = await getEmbedding(question);
  const { labels: indices } = await search(queryEmbedding, 3);
  const context = indices.map((i) => chunks[i]).join(" ");
  const answer = await generateAnswer(question, context);
  res.json({ answer });
});

app.listen(process.env.PORT, () =>
  console.log(`RAG demo running on http://localhost:${process.env.PORT}`)
);
