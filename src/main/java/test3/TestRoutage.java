package test3;

import assistant.Assistant;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

public class TestRoutage {

    public static void main(String[] args) throws Exception {

        // --- LLM ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --- Ingestion des documents ---
        EmbeddingStore<TextSegment> storeIA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeGL = new InMemoryEmbeddingStore<>();

        ingestPdf("rag.pdf", storeIA, embeddingModel);
        ingestPdf("Support.pdf", storeGL, embeddingModel);

        // --- 2 retrievers ---
        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(storeIA)
                .maxResults(3)
                .minScore(0.35)
                .build();

        ContentRetriever retrieverGL = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(storeGL)
                .maxResults(3)
                .minScore(0.35)
                .build();


        // --- DESCRIPTION RÉELLE DES PDF ---
        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();

        descriptions.put(retrieverIA,
                """
                Documents sur l'intelligence artificielle : RAG, embeddings, LLM, ingestion, retrieval.
                Contenu : phases du RAG, vecteurs, modèles d'embeddings, pipeline IA.
                """);

        descriptions.put(retrieverGL,
                """
                Documents sur le Génie Logiciel et la Qualité Logicielle.
                Contenu : qualité logicielle, ISO/IEC 25010, AQL, tests, maintenance, SOLID, désastres logiciels.
                """);

        // --- ROUTER ---
        LanguageModelQueryRouter router = new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = dev.langchain4j.service.AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        // --- Boucle utilisateur ---
        Scanner scanner = new Scanner(System.in);
        System.out.println("Test 3 - Routage activé. Pose une question :");

        while (true) {
            System.out.print("\nVous > ");
            String q = scanner.nextLine();
            if (q == null || q.equalsIgnoreCase("exit")) break;

            System.out.println("\nAssistant > " + assistant.chat(q));
        }
    }

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
                TestRoutage.class.getClassLoader().getResource(name)
        ).toURI());
    }
}
