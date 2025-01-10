const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const os = require('os');

module.exports = { detectDevices, runModel, objectDetection }

// Sharp settings
sharp.cache(100);  // Increased cache size
sharp.concurrency(2);  // Allow 2 concurrent operations since we're using Promise.all
sharp.simd(true);  // Ensure SIMD optimizations are enabled

// GLOBAL VARIABLES
// OpenVINO:
const core = new ov.Core();
if (core.getAvailableDevices().includes('CPU')) {
    core.setProperty("CPU", {
        "INFERENCE_NUM_THREADS": Math.max(2, os.cpus().length - 1),
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_HYPER_THREADING": "YES"
    });
}
const ovModels = new Map(); // compiled models
let model = null; // read model
const watermarkCache = new Map();
let baseWatermark = null;
let isInitialized = false;

const inputSize = { w: 640, h: 480 };
let batchno_classid_score_x1y1x2y2 = null;

const preprocessBuffer = new Float32Array(inputSize.w * inputSize.h * 3);
const normalizedBuffer = new Float32Array(inputSize.w * inputSize.h * 6);
const inferRequests = new Map();

async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

async function getModelPath() {
    if (fs.existsSync(path.join(__dirname, '../../app.asar'))){
        //if running compiled program
        return path.join(__dirname, "../../app.asar.unpacked/models/yolov9_n_wholebody25_post_0100_1x3x480x640.xml");
    } else {
        //if running npm start
    return path.join(__dirname, "../models/yolov9_n_wholebody25_post_0100_1x3x480x640.xml");
    }
}

async function getModel(device) {
    // if model not loaded
    if (model == null) {
        const modelPath = await getModelPath();
        model = await core.readModel(modelPath);
    }

    // if cached
    if (ovModels.has(device)) return ovModels.get(device)

    // compile and cache
    let compiledModel = await core.compileModel(model, device);
    ovModels.set(device, compiledModel);

    return compiledModel;
}


function normalizeArray(array) {
    const throughput = 0.5;
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < array.length; i++) {
        const val = array[i];
        if (val < min) min = val;
        if (val > max) max = val;
    }

    if (max === min) {
        normalizedBuffer.fill(0);
        return normalizedBuffer;
    }

    for (let i = 0; i < array.length; i++) {
        const coef = (array[i] - min) / (max - min);
        normalizedBuffer[i] = coef > throughput ? 1 : 0;
    }

    return normalizedBuffer;
}


async function preprocess(originalImg) {
    // 1) 画像を [H, W, 3] の形 (RGB) で取得
    const inputImg = await originalImg
        .resize(inputSize.w, inputSize.h, { fit: 'fill' })
        .removeAlpha() // αチャネルを除去 → 3チャネル(RGB)
        .raw()         // ピクセルデータをRGBA(→RGB)の生配列として取得
        .toBuffer();

    // 2) 出力バッファを [N, C, H, W] = [1, 3, height, width] サイズで用意
    //    ここでは型を float32 (Float32Array) として用意する例
    const { w, h } = inputSize;
    const outChannels = 3;
    const outSize = 1 * outChannels * h * w;
    const preprocessBuffer = new Float32Array(outSize);

    // 3) HWC(RGB) → CHW(BGR) への変換
    //    (row, col)ピクセルに対して:
    //      inIndex  = (row * width + col) * 3        // [R, G, B]
    //      outIndexB= 0 * (h*w) + (row * w) + col    // C=0 (B)
    //      outIndexG= 1 * (h*w) + (row * w) + col    // C=1 (G)
    //      outIndexR= 2 * (h*w) + (row * w) + col    // C=2 (R)
    for (let row = 0; row < h; row++) {
        for (let col = 0; col < w; col++) {
            const inIndex = (row * w + col) * 3;

            const R = inputImg[inIndex + 0];
            const G = inputImg[inIndex + 1];
            const B = inputImg[inIndex + 2];

            // BGR の順に出力 (N=0 は省略)
            const outIndexB = 0 * (h*w) + row * w + col;
            const outIndexG = 1 * (h*w) + row * w + col;
            const outIndexR = 2 * (h*w) + row * w + col;

            preprocessBuffer[outIndexB] = B;
            preprocessBuffer[outIndexG] = G;
            preprocessBuffer[outIndexR] = R;
        }
    }

    // 4) ov.Tensor を NCHW: [1, 3, H, W] の形で生成 (要素型: f32)
    const shape = [1, 3, h, w];
    return new ov.Tensor(ov.element.f32, shape, preprocessBuffer);
}

function postprocess(resultTensor) {
    // outputData を 2次元配列に変換 (rows=N, cols=?)
    // 例: 1推論に対して Nx7 個の検出結果が返る想定
    //console.time('postprocess');
    const dimD = 7; // 例: [batchNo, classId, score, x1, y1, x2, y2] の7次元と仮定
    const rowCount = Math.floor(resultTensor.length / dimD);
    const batchno_classid_score_x1y1x2y2s = [];

    const threshold = 0.35; // 検出スコアしきい値
    const excludedIds = new Set([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23]); // 検出除外するクラスID

    for (let row = 0; row < rowCount; row++) {
        const startIdx = row * dimD;
        // [batchNo, classId, score, x1, y1, x2, y2] として取り出す
        batchno_classid_score_x1y1x2y2s.push([
            resultTensor[startIdx + 0], // batchNo
            resultTensor[startIdx + 1], // classId
            resultTensor[startIdx + 2], // score
            resultTensor[startIdx + 3], // x1
            resultTensor[startIdx + 4], // y1
            resultTensor[startIdx + 5], // x2
            resultTensor[startIdx + 6]  // y2
        ]);
    }

    // スコアが threshold を超えるものだけフィルタ
    const filteredData = batchno_classid_score_x1y1x2y2s.filter((row) => {
        const score = row[2];
        return score >= threshold;
    });

    // excludedIds でフィルタ
    const boxesData = filteredData.filter((row) => {
        const classId = row[1];
        return !excludedIds.has(classId);
    });

    //console.timeEnd('postprocess');
    return boxesData;
}

async function getInferRequest(device) {
    if (inferRequests.has(device)) {
        return inferRequests.get(device);
    }
    const model = await getModel(device);
    const inferRequest = model.createInferRequest();
    inferRequests.set(device, inferRequest);
    return inferRequest;
}


async function runModel(img, width, height, device) {
    const originalImg = sharp(img.data, { raw: { channels: 4, width, height } });
    const inputTensor = await preprocess(originalImg);
    const inferRequest = await getInferRequest(device);
    const startTime = performance.now();

    //console.time('inference');
    inferRequest.setInputTensor(inputTensor);
    inferRequest.infer();
    const outputLayer = (await getModel(device)).outputs[0];
    const resultTensor = inferRequest.getTensor(outputLayer);
    //console.timeEnd('inference');

    console.log('det: ', resultTensor.data);
    const stopTime = performance.now();
    const inferenceTime = (stopTime - startTime);

    console.log(`##### inferenceTime: ${inferenceTime} ms`);

    batchno_classid_score_x1y1x2y2s = postprocess(resultTensor);

    return {
        width: width,
        height: height,
        inferenceTime: inferenceTime
    };
}

async function objectDetection(image, width, height) {
    // 検出結果が null あるいは空なら元画像のみ返す
    if (!batchno_classid_score_x1y1x2y2s || batchno_classid_score_x1y1x2y2s.length === 0) {
        return {
            img: image.data,
            width: width,
            height: height
        };
    }

    console.log('det: ', batchno_classid_score_x1y1x2y2s);

    // classId が 25 クラスある想定
    // 好みの配色を 25 個セットしておく (例として使用)
    const colorPalette = [
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
        '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080',
        '#FFFFFF', '#e2228b', '#22e28b', '#262e9b', '#888888'
    ];

    try {
        // ---- (1) バウンディングボックス用 SVG を作成 ----
        const strokeWidth = 2; // 線の太さなど好みで変更

        // <rect> 要素の配列を作成
        const rects = batchno_classid_score_x1y1x2y2s.map(det => {
            const [batchNo, classId, score, x1, y1, x2, y2] = det;

            // classId が 25 以上の可能性があるなら、念のため mod を取る
            const strokeColor = colorPalette[classId % colorPalette.length];

            // 幅と高さ
            const w = x2 - x1;
            const h = y2 - y1;

            // 描画する <rect> と、あわせて文字情報を描画 (例: score 表示)
            return `
                <rect
                    x="${x1}"
                    y="${y1}"
                    width="${w}"
                    height="${h}"
                    fill="none"
                    stroke="${strokeColor}"
                    stroke-width="${strokeWidth}"
                />
                <!-- score やクラスIDなどを表示したい場合は <text> 要素も追加 -->
                <text
                    x="${x1}"
                    y="${Math.max(y1 - 5, 0)}"  /* 矩形の上あたりに表示 */
                    fill="${strokeColor}"
                    font-size="16"
                    font-weight="bold"
                    stroke="#000"         /* 文字のフチ取り */
                    stroke-width="0.5"
                    paint-order="stroke"
                >
                    class: ${classId}, score: ${score.toFixed(2)}
                </text>
                `;
        }).join('');

        // SVG 全体を文字列で定義
        const svgOverlay = `
            <svg
            width="${width}"
            height="${height}"
            viewBox="0 0 ${width} ${height}"
            xmlns="http://www.w3.org/2000/svg"
            >
            ${rects}
            </svg>
        `;

        // ---- (2) Sharp で元画像に SVG を合成 ----
        const imageWithBoxes = await sharp(image.data, {
            raw: {
            channels: 4,
            width: width,
            height: height
            },
            limitInputPixels: false
        })
            .composite([
            {
                input: Buffer.from(svgOverlay),
                top: 0,
                left: 0
            }
            ])
            .raw()
            .toBuffer();

        // 出力用に Uint8ClampedArray に変換
        return {
            img: new Uint8ClampedArray(imageWithBoxes),
            width,
            height
        };

    } catch (error) {
        console.error('Error in bounding box drawing:', error);
        // エラー発生時は元データを返却
        return {
            img: image.data,
            width,
            height
        };
    }
}
