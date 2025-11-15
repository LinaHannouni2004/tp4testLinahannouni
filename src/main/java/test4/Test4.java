package test4;


import assistant.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.input.PromptTemplate;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.Query;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Test4 {

    public static void main(String[] args) throws Exception {

        // --- 1) MODELE LLM ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // --- 2) MODELE D’EMBEDDING ---
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --- 3) INGESTION DU PDF rag.pdf UNIQUEMENT ---
        EmbeddingStore<TextSegment> storeIA = new InMemoryEmbeddingStore<>();
        ingestPdf("rag.pdf", storeIA, embeddingModel);

        // --- 4) CREATION DU RETRIEVER ---
        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(storeIA)
                .maxResults(3)
                .minScore(0.35)
                .build();

        // --- 5) TEMPLATE POUR LE BONUS ---
        PromptTemplate template = PromptTemplate.from(
                """
                Est-ce que la requête suivante porte sur l'intelligence artificielle, 
                le RAG ou le Fine-Tuning ?

                Requête utilisateur : "{{question}}"

                Réponds strictement par : "oui", "non" ou "peut-être".
                """
        );

        // --- 6) QUERY ROUTER PERSONNALISÉ ---
        QueryRouter customRouter = new QueryRouter() {
            @Override
            public List<ContentRetriever> route(Query query) {

                // injecter la question dans le template
                String prompt = template.apply(Map.of("question", query.text())).text();

                // poser la question AU LLM directement
                String answer = model.chat(prompt);

                String rep = answer.toLowerCase(Locale.ROOT);

                if (rep.contains("non")) {
                    return Collections.emptyList();   // PAS DE RAG
                } else {
                    return Collections.singletonList(retrieverIA); // UTILISER RAG
                }
            }
        };

        // --- 7) RETRIEVAL AUGMENTOR ---
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(customRouter)
                .build();

        // --- 8) ASSISTANT ---
        Assistant assistant = dev.langchain4j.service.AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        // --- 9) BOUCLE UTILISATEUR ---
        Scanner scanner = new Scanner(System.in);
        System.out.println("Test 4 - Pas de RAG. Pose une question :");

        while (true) {
            System.out.print("\nVous > ");
            String q = scanner.nextLine();
            if (q == null || q.equalsIgnoreCase("exit")) break;

            System.out.println("\nAssistant > " + assistant.chat(q));
        }
    }

    // --------------------------- OUTILS -----------------------------

    private static void ingestPdf(String fileName,
                                  EmbeddingStore<TextSegment> store,
                                  EmbeddingModel embeddingModel) throws Exception {

        Path path = getResourcePath(fileName);
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
        var splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(doc);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        store.addAll(embeddings, segments);
    }

    private static Path getResourcePath(String name) throws URISyntaxException {
        return Paths.get(Objects.requireNonNull(
                Test4.class.getClassLoader().getResource(name)
        ).toURI());
    }
}

