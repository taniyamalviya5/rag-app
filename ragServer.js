import xlsx from "xlsx";
import * as faiss from "faiss-node";
import { pipeline } from "@xenova/transformers";

const { readFile, utils } = xlsx;

const readExcel = (filePath) => {
  const workbook = readFile(filePath);
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];
  return utils.sheet_to_json(sheet); // returns array of rows
};

// Example
const data = readExcel("./document/file_example_XLS_10.xls");

const chunkText = (text, chunkSize = 1000, overlap = 100) => {
  const chunks = [];
  let start = 0;
  while (start < JSON.stringify(text).length) {
    const end = Math.min(start + chunkSize, JSON.stringify(text).length);
    chunks.push(JSON.stringify(text).slice(start, end));
    start += chunkSize - overlap;
  }
  return chunks;
};

// Example
export const chunks = data.map((row) => chunkText(row || ""));
console.log("Chunks:", chunks.length);

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

export const getEmbedding = async (text) => {
  //   const res = await fetch(
  //     "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
  //     {
  //       method: "POST",
  //       headers: {
  //         Authorization: `Bearer ${HF_TOKEN}`,
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify({ inputs: text }),
  //     }
  //   );
  //   const json = await res.json();
  //   return json[0]; // returns vector
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  const result = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(result.data);
};

// Generate embeddings for all chunks
const embeddings = await Promise.all(
  chunks.map((chunk) => getEmbedding(chunk))
);

const dim = embeddings[0].length; // 32

const index = new faiss.default.IndexFlatL2(dim);
const flat = [].concat(...embeddings); // flatten to 1D JS array

index.add(flat);

// Search function
export const search = (queryVec, k = 3) => {
  const results = index.search(Array.from(queryVec), k);
  return results; // returns distances + indices
};

export const generateAnswer = async (query, context) => {
  const generator = await pipeline("text-generation", "Xenova/distilgpt2"); // free local
  const prompt = `Answer the question using the context:\nContext: ${context}\nQuestion: ${query}`;
  const output = await generator(prompt, { max_length: 200 });
  return output[0].generated_text;
};

// Full flow
// const query = "What is the Country of Mara?";
// const queryEmbedding = await getEmbedding(query);
// const { labels: indices } = search(queryEmbedding, 3);
// const context = indices.map((i) => chunks[i]).join(" ");
// const answer = await generateAnswer(query, context);

// console.log("Answer:", answer);
