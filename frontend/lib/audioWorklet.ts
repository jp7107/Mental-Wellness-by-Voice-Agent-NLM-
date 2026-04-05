// AudioWorkletProcessor — runs on dedicated audio thread
// Converts Float32 microphone samples to PCM16 and posts to main thread

const PROCESSOR_CODE = `
class MindEaseProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._targetSamples = ${16000 * 2}; // 2 seconds at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, mono
    for (let i = 0; i < samples.length; i++) {
      this._buffer.push(samples[i]);
    }

    if (this._buffer.length >= this._targetSamples) {
      // Convert Float32 to Int16 PCM
      const pcm = new Int16Array(this._buffer.length);
      for (let i = 0; i < this._buffer.length; i++) {
        const s = Math.max(-1, Math.min(1, this._buffer[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage({ pcm16: pcm.buffer }, [pcm.buffer]);
      this._buffer = [];
    }
    return true;
  }
}
registerProcessor('mindease-processor', MindEaseProcessor);
`;

export function createWorkletBlobURL(): string {
  const blob = new Blob([PROCESSOR_CODE], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}
