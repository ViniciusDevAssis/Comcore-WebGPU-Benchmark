// A mesma versao de Transformers.js e o mesmo checkpoint da demo da WebML Community.
// Import remoto permite publicar estes tres arquivos sem npm ou processo de build.
import {
  AutoProcessor,
  Qwen3_5ForConditionalGeneration,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const $ = (id) => document.getElementById(id);
const MODEL_IDS = {
  "onnx-community/Qwen3.5-2B-ONNX-OPT": "Qwen3.5-2B",
  "onnx-community/Qwen3.5-0.8B-ONNX-OPT": "Qwen3.5-0.8B",
};
const PRESETS = {
  short: "Explique em português brasileiro, em duas frases, o que é uma API REST.",
  dialog: "Explique, em português brasileiro, três cuidados ao criar senhas fortes, usando linguagem simples e no máximo quatro frases.",
};

let processor = null;
let model = null;
let busy = false;
let adapterData = null;
const results = [];

function formatSeconds(milliseconds) {
  return Number.isFinite(milliseconds) ? `${(milliseconds / 1000).toFixed(2).replace('.', ',')} s` : "—";
}
function formatRate(value) {
  return Number.isFinite(value) ? `${value.toFixed(2).replace('.', ',')} tok/s` : "—";
}
function showError(message) {
  $("error").textContent = String(message || "Erro desconhecido");
  $("error").hidden = false;
}
function clearError() {
  $("error").hidden = true;
  $("error").textContent = "";
}
function setBusy(value) {
  busy = value;
  $("load").disabled = value || !adapterData?.supported || Boolean(model);
  $("model").disabled = value || Boolean(model);
  $("run").disabled = value || !model;
  $("max-tokens").disabled = value;
}

async function inspectGpu() {
  $("platform").textContent = navigator.userAgentData?.platform || navigator.platform || "Desconhecida";
  $("memory").textContent = navigator.deviceMemory ? `~${navigator.deviceMemory} GB (estimativa)` : "Não informado";
  if (!globalThis.isSecureContext) {
    $("webgpu").textContent = "HTTPS ou localhost necessário";
    $("gpu").textContent = "Indisponível";
    $("model-state").textContent = "Publique este teste por HTTPS.";
    return;
  }
  if (!navigator.gpu) {
    $("webgpu").textContent = "Indisponível no navegador";
    $("gpu").textContent = "Indisponível";
    $("model-state").textContent = "O navegador não oferece WebGPU.";
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Nenhum adaptador WebGPU disponível.");
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    adapterData = {
      supported: true,
      vendor: info.vendor || "Não identificado",
      architecture: info.architecture || "Não identificada",
      device: info.device || "",
      fp16: adapter.features.has("shader-f16"),
    };
    $("webgpu").textContent = "Disponível";
    $("gpu").textContent = `${adapterData.vendor} / ${adapterData.architecture}`;
    $("fp16").textContent = adapterData.fp16 ? "Sim (shader-f16)" : "Não";
    $("model-state").textContent = "Pronto para carregar. Selecione o modelo.";
    $("load").disabled = false;
  } catch (error) {
    $("webgpu").textContent = "Falha ao inicializar";
    $("gpu").textContent = error.message;
    $("model-state").textContent = "Não foi possível obter um adaptador WebGPU.";
  }
}

$("preset").addEventListener("change", () => {
  const preset = $("preset").value;
  if (PRESETS[preset]) $("prompt").value = PRESETS[preset];
});
$("prompt").addEventListener("input", () => { $("preset").value = "custom"; });

$("load").addEventListener("click", async () => {
  if (busy || model || !adapterData?.supported) return;
  clearError();
  setBusy(true);
  const id = $("model").value;
  const t0 = performance.now();
  $("progress").hidden = false;
  $("progress").value = 0;
  $("progress-num").textContent = "";
  try {
    $("model-state").textContent = "Carregando processador e tokenizador…";
    processor = await AutoProcessor.from_pretrained(id);
    $("model-state").textContent = "Baixando pesos e inicializando WebGPU…";
    model = await Qwen3_5ForConditionalGeneration.from_pretrained(id, {
      dtype: {
        embed_tokens: "q4",
        vision_encoder: "fp16",
        decoder_model_merged: "q4",
      },
      device: "webgpu",
      progress_callback: (p) => {
        // Vários arquivos são carregados; a porcentagem exibida é do arquivo atual.
        if (p?.status === "progress" && typeof p.progress === "number") {
          $("progress").value = Math.max(0, Math.min(100, p.progress));
          $("progress-num").textContent = `${Math.round(p.progress)}% (arquivo atual)`;
        } else if (p?.status === "initiate" && p.file) {
          $("model-state").textContent = `Carregando ${String(p.file).split('/').at(-1)}`;
        }
      },
    });
    $("progress").hidden = true;
    $("progress-num").textContent = "";
    $("model-state").textContent = `Modelo pronto. Tempo de carga: ${formatSeconds(performance.now() - t0)}.`;
    $("run-state").textContent = `${MODEL_IDS[id]} carregado`;
  } catch (error) {
    console.error(error);
    model = null;
    processor = null;
    $("model-state").textContent = "Falha no carregamento. Tente reiniciar a aba ou o modelo 0.8B.";
    $("progress").hidden = true;
    showError(`Falha ao carregar: ${error?.message || error}`);
  } finally {
    setBusy(false);
  }
});

$("run").addEventListener("click", async () => {
  if (busy || !model || !processor) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return showError("Informe um prompt para o teste.");
  clearError();
  setBusy(true);
  const maxNewTokens = Number($("max-tokens").value);
  $("response").textContent = "Preparando entrada…";
  $("run-state").textContent = "Preparando prompt";

  let tokenCount = 0;
  let firstTokenAt = null;
  let textGenerated = "";
  let startedAt = null;
  try {
    // O mesmo template nao-thinking utilizado na demonstracao WebML.
    // Uma unica mensagem por teste: sem cache de conversa anterior.
    const formatted = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
    const encodeStartedAt = performance.now();
    const inputs = await processor(formatted);
    const encodeMs = performance.now() - encodeStartedAt;
    $("response").textContent = "";
    $("run-state").textContent = "Gerando resposta…";

    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => {
        if (firstTokenAt === null) firstTokenAt = performance.now();
        tokenCount++;
      },
      callback_function: (piece) => {
        textGenerated += piece;
        $("response").textContent = textGenerated || "Gerando…";
      },
    });
    startedAt = performance.now();
    // A saida nao e armazenada fora desta pagina; os resultados exportados sao so metricas.
    await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      streamer,
    });
    const finishedAt = performance.now();
    const totalMs = finishedAt - startedAt;
    const ttftMs = firstTokenAt === null ? null : firstTokenAt - startedAt;
    const decodeMs = firstTokenAt === null ? null : finishedAt - firstTokenAt;
    // Taxa apos o 1o token; para testes com somente 1 token, a taxa nao e definida.
    const tps = tokenCount > 1 && decodeMs > 0 ? ((tokenCount - 1) * 1000) / decodeMs : null;
    $("ttft").textContent = formatSeconds(ttftMs);
    $("tps").textContent = formatRate(tps);
    $("total").textContent = formatSeconds(totalMs);
    $("tokens").textContent = String(tokenCount);
    $("run-state").textContent = "Teste concluído";
    if (!textGenerated.trim()) $("response").textContent = "O modelo encerrou a geração sem texto visível.";

    results.push({
      at: new Date().toISOString(),
      model: $("model").value,
      max_new_tokens: maxNewTokens,
      prompt_characters: prompt.length,
      prompt_preset: $("preset").value,
      token_count_streamer: tokenCount,
      encode_ms: Number(encodeMs.toFixed(2)),
      ttft_ms: ttftMs === null ? null : Number(ttftMs.toFixed(2)),
      generation_ms: Number(totalMs.toFixed(2)),
      decode_ms: decodeMs === null ? null : Number(decodeMs.toFixed(2)),
      tokens_per_second_excluding_first: tps === null ? null : Number(tps.toFixed(2)),
      webgpu: adapterData,
      device_memory_gb_approx: navigator.deviceMemory || null,
      platform: navigator.userAgentData?.platform || navigator.platform || "unknown",
      user_agent: navigator.userAgent,
    });
    renderHistory();
  } catch (error) {
    console.error(error);
    $("run-state").textContent = "Erro na inferência";
    showError(`Falha na geração: ${error?.message || error}`);
  } finally {
    setBusy(false);
  }
});

function renderHistory() {
  $("history").replaceChildren();
  for (const r of results) {
    const tr = document.createElement("tr");
    const columns = [MODEL_IDS[r.model], String(r.prompt_characters) + " caracteres", formatSeconds(r.ttft_ms), formatRate(r.tokens_per_second_excluding_first), formatSeconds(r.generation_ms)];
    for (const value of columns) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    $("history").append(tr);
  }
  $("export").disabled = results.length === 0;
}

$("export").addEventListener("click", () => {
  if (!results.length) return;
  const json = JSON.stringify({
    benchmark: "ComCore WebGPU Browser Benchmark",
    runtime: "@huggingface/transformers@4.2.0",
    note: "Dados locais, sem prompt nem resposta. Teste experimental sem validacao clinica.",
    results,
  }, null, 2);
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `comcore-webgpu-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

inspectGpu();
