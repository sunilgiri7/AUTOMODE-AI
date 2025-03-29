<h1 align="center">AUTOMODE-AI</h1>
<h2 align="center">Enterprise AI Chatbot & Document Ingestion System</h2>

<p align="center">
  <strong>Advanced Chatbot Powered by LangChain, FAISS, and Reinforcement Learning.</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a> •
</p>

<hr>

<h2 id="features">Features</h2>

<ul>
  <li><strong>Chatbot Interface:</strong> A conversational UI built with HTML, CSS, and JavaScript (Socket.IO) for real-time interactions.</li>
  <li><strong>Document Upload & Ingestion:</strong> Secure file upload functionality (supports txt, pdf, doc, docx, md) to ingest documents into a FAISS vector store.</li>
  <li><strong>Advanced Retrieval:</strong> Combines vector-based search (via FAISS) with BM25 text retrieval and an ensemble retriever. Falls back gracefully if compression (via CohereRerank) is not available.</li>
  <li><strong>Contextual Compression:</strong> Uses a Cohere reranking model (with fallback) to compress and rerank document embeddings for improved answer quality.</li>
  <li><strong>Knowledge Graph Integration:</strong> Optionally integrates with Neo4j for graph-based question answering.</li>
  <li><strong>HyDE (Hypothetical Document Embeddings):</strong> Generates hypothetical answers to augment context and improve retrieval.</li>
  <li><strong>Reinforcement Learning:</strong> A continuous learning loop that updates the model based on user feedback.</li>
  <li><strong>Chat History & Analytics:</strong> Session-based chat logging with history formatting and accuracy calculation from user feedback.</li>
</ul>

<h2 id="installation">Installation</h2>

<p>Clone the repository and install the required dependencies:</p>

```bash
git clone https://github.com/yourusername/enterprise-ai-chatbot.git
cd enterprise-ai-chatbot
pip install -r requirements.txt
```


<p>The project relies on:</p>
<ul>
  <li>Python 3.12+</li>
  <li>Flask and Flask-SocketIO</li>
  <li>LangChain (with modules: <code>langchain_ollama</code>, <code>langchain_huggingface</code>, etc.)</li>
  <li>FAISS for vector indexing</li>
  <li>Neo4j (optional for knowledge graph integration)</li>
  <li>Additional libraries: <code>cohere</code>, <code>pydantic</code>, etc.</li>
</ul>
<h2 id="usage">Usage</h2>
<p>To start the Flask server with Socket.IO integration, run:</p>
<pre class="!overflow-visible"><div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none"></div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none px-4 py-1" aria-label="Copy"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg></button></span><span class="" data-state="closed"><button class="flex select-none items-center gap-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-xs"><path d="M2.5 5.5C4.3 5.2 5.2 4 5.5 2.5C5.8 4 6.7 5.2 8.5 5.5C6.7 5.8 5.8 7 5.5 8.5C5.2 7 4.3 5.8 2.5 5.5Z" fill="currentColor" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"></path><path d="M5.66282 16.5231L5.18413 19.3952C5.12203 19.7678 5.09098 19.9541 5.14876 20.0888C5.19933 20.2067 5.29328 20.3007 5.41118 20.3512C5.54589 20.409 5.73218 20.378 6.10476 20.3159L8.97693 19.8372C9.72813 19.712 10.1037 19.6494 10.4542 19.521C10.7652 19.407 11.0608 19.2549 11.3343 19.068C11.6425 18.8575 11.9118 18.5882 12.4503 18.0497L20 10.5C21.3807 9.11929 21.3807 6.88071 20 5.5C18.6193 4.11929 16.3807 4.11929 15 5.5L7.45026 13.0497C6.91175 13.5882 6.6425 13.8575 6.43197 14.1657C6.24513 14.4392 6.09299 14.7348 5.97903 15.0458C5.85062 15.3963 5.78802 15.7719 5.66282 16.5231Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path><path d="M14.5 7L18.5 11" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><span>python app.py</span></div></div></pre>


<p>The application will:</p>
<ul>
  <li>Load (or create) the FAISS vector store for document embeddings.</li>
  <li>Initialize the QA chain using LangChain with advanced retrieval and a custom prompt template.</li>
  <li>Expose endpoints for chat interactions (via Socket.IO) and file uploads.</li>
</ul>
<p>Open your browser and navigate to <code>http://your-server-ip:5000</code> to access the chatbot interface. Use the provided upload form to ingest documents, which will be processed and added to the vector store.</p>
<h2 id="architecture">Architecture Overview</h2>
<ul>
  <li>
    <strong>Backend (chatbot_backend.py):</strong>
    <ul>
      <li>Initializes the LLM (<code>OllamaLLM</code>) with specified parameters.</li>
      <li>Uses <code>HuggingFaceEmbeddings</code> to embed documents and queries.</li>
      <li>Builds an advanced retriever combining BM25 and vector search from FAISS.</li>
      <li>Constructs the QA chain via a pipeline using LangChain’s runnables, prompt templates, and output parsers.</li>
      <li>Integrates a reinforcement learning loop to update model weights based on user feedback.</li>
    </ul>
  </li>
  <li>
    <strong>Frontend (index.html):</strong>
    <ul>
      <li>A responsive, modern HTML/CSS/JS interface styled with Bootstrap.</li>
      <li>Real-time messaging using Socket.IO.</li>
      <li>File upload component for document ingestion.</li>
    </ul>
  </li>
  <li>
    <strong>Document Manager:</strong>
    <ul>
      <li>Handles file saving, binary file checking, and ingestion of valid documents into the FAISS vector store.</li>
    </ul>
  </li>
</ul>
<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please follow these steps:</p>
<ol>
  <li>Fork the repository.</li>
  <li>Create a new branch for your feature or bug fix.</li>
  <li>Make your changes and write tests if applicable.</li>
  <li>Submit a pull request with a clear description of your changes.</li>
</ol>
<hr>
<p align="center">
  <em>Developed with ❤️ by Sunil Giri -> LinkedIn(https://www.linkedin.com/in/sunil-giri77/) Github(https://github.com/sunilgiri7)</em>
</p>
