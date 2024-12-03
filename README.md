# ** Project - BenchBot **

## **Overview**
BenchBot is a knowledge-based assistant that combines the power of LangChain, OpenAI, Pinecone, and React for dynamic document retrieval and user interaction.

---

## <i>Backend Setup </i>

### **1. Install Python Dependencies**
Run the following commands to install the required all dependencies for the backend:

```bash
pip install -r requirements.txt
```
### 2. Set up Environment Variables

- Create a file named .env in your backend directory.
- Add the following line to the .env file:
```
        OPENAI_API_KEY=your_openai_api_key
```
- Replace your_openai_api_key with your actual OpenAI API key.


### 3. Running the Backend

Use the following command to start the backend application:
```bash
chainlit run app.py
```
This will start the Chainlit server, where you can interact with the assistant

## <i> Frontend Setup  </i>


### 1. Install React Dependencies
- Navigate to the frontend directory and install the required dependencies using the following commands:

```bash
cd frontend
npm install react react-dom react-scripts axios
```
- If you plan to use React Router or state management, you can also install these:

```bash
npm install react-router-dom
npm install @reduxjs/toolkit react-redux
```

### 2. Start the Development Server
- Run the following command to start the React development server:

```bash
npm start
```
The application will be accessible at http://localhost:3000

## <i> Development Tools </i>

### FAISS
- FAISS is used for efficient similarity search and clustering in the backend. It is integrated to optimize vector search performance.

Install FAISS using:

```bash
pip install faiss-cpu
```

- For GPU support, install:

```bash
pip install faiss-gpu
```

### Pinecone
- Pinecone is used as a vector database for scalable, low-latency search and retrieval.

Install the Pinecone client using:
```bash
pip install pinecone-client
```

## <i>Blocks of code </i>

```
BACKEND/
├── BenchBot/
│   ├── app.py                # Backend Python code
│   ├── pinecone.py           # Backend Python code
│   ├── requirements.txt      # backend dependencies
│   ├── .env                  # Environment variables file
│   ├── README.md                 # Project documentation
|   ├── frontend/
|       ├── benchBot/
│            ├── src/
│            │   ├── components/   # React components
|            |   ├── common /      # Navigation, Foooter
│            │   ├── App.js        # Main React app entry
|            |   └── ...
│            ├── package.json      # Frontend dependencies   
│            └── ...
└── ...

```

## How to Contribute
- Clone the repository.
```bash
git clone https/shh url 
```
- Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```
- Commit your changes:
```bash
git commit -m "Add your message here"
```
- Push to the branch:
```bash
git push origin feature/your-feature-name
```
- Create a pull request.


## <i>end ... </i>
