# ComCore — Benchmark WebGPU (Qwen3.5-2B)

Benchmark experimental de **inferência local no navegador** (Chrome Android ou desktop), sem backend, Vite, Node ou compilação. Os três arquivos `index.html`, `main.js` e `style.css` formam o site.

## Modelo e runtime

- Modelos: `onnx-community/Qwen3.5-2B-ONNX-OPT` (principal) e `onnx-community/Qwen3.5-0.8B-ONNX-OPT` (controle).
- Biblioteca: `@huggingface/transformers@4.2.0` (via jsDelivr, mesma versão presente na demo da WebML Community consultada em 20/10/2026).
- Execução: WebGPU do navegador. **Não utiliza GGUF nem llama.cpp**. A inferência é local depois do download dos arquivos; CDN e Hugging Face são usados para distribuir código e pesos.
- Quantização: ONNX `embed_tokens=q4`, `decoder_model_merged=q4`, `vision_encoder=fp16` (mesma configuração da demonstração). A torre de visão pode consumir memória extra mesmo em prompts só de texto.
- Prompt fixado em modo sem raciocínio explícito (`<think>\n\n</think>`), com entradas independentes para comparabilidade.

## Como publicar por HTTPS (recomendado para celular)

1. Crie um repositório GitHub público novo, por exemplo `comcore-webgpu-benchmark`.
2. Extraia este ZIP e envie os três arquivos diretamente à **raiz** do repositório, não a pasta que os contém.
3. Em **Settings → Pages → Build and deployment**, escolha **Deploy from a branch** e selecione `main` / `(root)`; salve.
4. Abra o endereço que o GitHub Pages apresentar em `https://<usuario>.github.io/comcore-webgpu-benchmark/`.
5. No Edge 40 Neo, abra o link no Chrome, selecione **Qwen3.5-2B**, toque **Carregar modelo no dispositivo** e execute os prompts predefinidos.
6. Faça 3 testes seguidos e exporte o JSON. Registre também se o aparelho aqueceu, fechou a aba ou ficou sem memória.

Também é possível usar outro serviço de hospedagem estática por HTTPS. Abrir o arquivo HTML como `file://` não é recomendado: WebGPU exige contexto seguro e alguns imports podem ser bloqueados.

## O que significam os indicadores

- **TTFT:** tempo entre iniciar `generate()` e receber o primeiro token, incluindo prefill; não inclui o download e a preparação do prompt.
- **tokens/s:** `(tokens - 1) / tempo entre o primeiro token e o fim`, aproximação de velocidade de decodificação.
- **tempo total:** apenas tempo de execução de `generate()`.
- **encode_ms:** tempo gasto na preparação do prompt, exportado no JSON.
- **token_count_streamer:** tokens reportados pelo callback, que podem incluir um token final especial; comparar sempre com o mesmo runtime.
- **Carregamento:** exibido na interface; inclui download e inicialização, mas a ferramenta não diferencia cache aquecido de primeiro download.

## Limitações e segurança

- **O código não foi executado em um Edge 40 Neo nesta entrega**. A sintaxe JavaScript/HTML pode ser validada offline, mas o carregamento dos pesos e o comportamento WebGPU precisam ser testados no dispositivo real.
- GPUs e navegadores podem travar por falta de memória, especialmente com a versão 2B. Teste também 0.8B se necessário.
- O indicador `navigator.deviceMemory` **não** mede VRAM nem memória livre.
- Os dados exportados são locais e não incluem texto do usuário/saída. Não submeta informações pessoais, dados de saúde ou conversas reais.
- O benchmark mede desempenho, **não** avalia adequação clínica, segurança de triagem ou qualidade em situações de crise.
- Modelos e recursos são baixados de terceiros (Hugging Face e jsDelivr). Confirme os termos e políticas antes de utilização real.

## Referências

- Demo WebML Community: https://huggingface.co/spaces/webml-community/Qwen3.5-WebGPU
- Fonte da implementação base: https://huggingface.co/spaces/webml-community/Qwen3.5-WebGPU/blob/main/index.js
- Checkpoint: https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT
- Streaming Transformers.js: https://huggingface.co/docs/transformers.js/api/generation/streamers
