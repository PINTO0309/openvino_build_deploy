const { addon: ov } = require('openvino-node');
const { performance } = require('perf_hooks');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const os = require('os');

module.exports = { detectDevices, runModel, objectDetection };

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

// 推論する入力画像サイズ (640×480)
const inputSize = { w: 640, h: 480 };

// 推論結果を一時的に保持する配列 ([batchNo, classId, score, x1, y1, x2, y2] の配列)
let batchno_classid_score_x1y1x2y2s = [];

// バッファは毎回使い回す場合などに使う想定だが、ここでは単なる例示
const preprocessBuffer = new Float32Array(inputSize.w * inputSize.h * 3);
const normalizedBuffer = new Float32Array(inputSize.w * inputSize.h * 6);

// InferRequest のキャッシュ (デバイスごとに使い回し)
const inferRequests = new Map();


//---------------------------------------------------------------------
// デバイス一覧の取得
//---------------------------------------------------------------------
async function detectDevices() {
    return ["AUTO"].concat(core.getAvailableDevices());
}

//---------------------------------------------------------------------
// モデルファイルのパスを取得
//---------------------------------------------------------------------
async function getModelPath() {
    if (fs.existsSync(path.join(__dirname, '../../app.asar'))){
        //if running compiled program
        return path.join(__dirname, "../../app.asar.unpacked/models/yolov9_n_wholebody25_post_0100_1x3x480x640.xml");
    } else {
        //if running npm start
        return path.join(__dirname, "../models/yolov9_n_wholebody25_post_0100_1x3x480x640.xml");
    }
}

//---------------------------------------------------------------------
// モデルの読み込みおよびコンパイル (デバイスごとにキャッシュ)
//---------------------------------------------------------------------
async function getModel(device) {
    // まだモデルを readModel していない場合
    if (model == null) {
        const modelPath = await getModelPath();
        model = await core.readModel(modelPath);
    }
    // 既に同デバイスでコンパイル済みなら再利用
    if (ovModels.has(device)) return ovModels.get(device);

    // コンパイル → キャッシュ
    let compiledModel = await core.compileModel(model, device);
    ovModels.set(device, compiledModel);
    return compiledModel;
}

//---------------------------------------------------------------------
// 推論前の前処理 (Sharp で 640×480 RGB画像を取得 → Float32 CHW(BGR) テンソルに変換)
//---------------------------------------------------------------------
async function preprocess(originalImg) {
    // 1) 画像を [H, W, 3] の形 (RGB) で取得
    //    → 640×480 にリサイズ & αチャネル除去
    const inputImg = await originalImg
        .resize(inputSize.w, inputSize.h, { fit: 'fill' })
        .removeAlpha() // αチャネルを除去 → 3チャネル(RGB)
        .raw()         // ピクセルデータを [R, G, B] の生配列として取得
        .toBuffer();

    // 2) 出力バッファを [N, C, H, W] = [1, 3, height, width] サイズで用意 (Float32Array)
    const { w, h } = inputSize;
    const outChannels = 3;
    const outSize = 1 * outChannels * h * w;
    const preprocessBuffer = new Float32Array(outSize);

    // 3) HWC(RGB) → CHW(BGR) への変換
    for (let row = 0; row < h; row++) {
        for (let col = 0; col < w; col++) {
            const inIndex = (row * w + col) * 3;

            const R = inputImg[inIndex + 0];
            const G = inputImg[inIndex + 1];
            const B = inputImg[inIndex + 2];

            // BGR の順に格納
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

//---------------------------------------------------------------------
// 推論結果の後処理 (出力テンサーを [batchNo, classId, score, x1, y1, x2, y2] の配列にまとめ、フィルタリング)
//---------------------------------------------------------------------
function postprocess(resultTensor) {
    const data = resultTensor.data;
    const dimD = 7; // [batchNo, classId, score, x1, y1, x2, y2] の7次元と想定
    const rowCount = Math.floor(data.length / dimD);

    // 閾値・除外クラス
    const threshold = 0.35;
    const excludedIds = new Set([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23]);

    // 取得した推論結果をまとめる
    const detected = [];
    for (let row = 0; row < rowCount; row++) {
        const startIdx = row * dimD;
        const batchNo = data[startIdx + 0];
        const classId = data[startIdx + 1];
        const score   = data[startIdx + 2];
        const x1      = data[startIdx + 3];
        const y1      = data[startIdx + 4];
        const x2      = data[startIdx + 5];
        const y2      = data[startIdx + 6];
        detected.push([batchNo, classId, score, x1, y1, x2, y2]);
    }

    // スコアがしきい値以上のものだけ
    const filteredData = detected.filter(([b, cid, sc]) => sc >= threshold);

    // 除外クラスでさらにフィルタ
    const boxesData = filteredData.filter(([b, cid]) => !excludedIds.has(cid));

    return boxesData;
}

//---------------------------------------------------------------------
// InferRequest の取得 (デバイスごとにキャッシュ)
//---------------------------------------------------------------------
async function getInferRequest(device) {
    if (inferRequests.has(device)) {
        return inferRequests.get(device);
    }
    const model = await getModel(device);
    const inferRequest = model.createInferRequest();
    inferRequests.set(device, inferRequest);
    return inferRequest;
}

//---------------------------------------------------------------------
// 画像データ (raw形式) を入力として推論を実行する
//---------------------------------------------------------------------
async function runModel(img, width, height, device) {
    // Sharp で "width×height (4チャネル)" の生画像を取り込み
    const originalImg = sharp(img.data, {
        raw: {
            channels: 4,
            width: width,
            height: height
        }
    });

    // 前処理 → 推論
    const inputTensor = await preprocess(originalImg);
    const inferRequest = await getInferRequest(device);

    const startTime = performance.now();
    inferRequest.setInputTensor(inputTensor);
    inferRequest.infer();
    const outputLayer = (await getModel(device)).outputs[0];
    const resultTensor = inferRequest.getTensor(outputLayer);
    const stopTime = performance.now();

    const inferenceTime = (stopTime - startTime);
    console.log(`##### inferenceTime: ${inferenceTime} ms`);

    // 後処理 → 結果をグローバル変数に保存 (本来は適宜 return などで渡すのが望ましい)
    batchno_classid_score_x1y1x2y2s = postprocess(resultTensor);

    return {
        width: width,
        height: height,
        inferenceTime: inferenceTime
    };
}

//---------------------------------------------------------------------
// バウンディングボックスを描画した画像を生成して返す
//---------------------------------------------------------------------
async function objectDetection(image, width, height) {
    // 推論結果が空ならそのまま返す
    // console.log('det: ', batchno_classid_score_x1y1x2y2s);
    if (!batchno_classid_score_x1y1x2y2s || batchno_classid_score_x1y1x2y2s.length === 0) {
        return {
            img: image.data,
            width: width,
            height: height
        };
    }

    // console.log('det: ', batchno_classid_score_x1y1x2y2s);

    // クラスIDが 25 クラスある想定で色を準備 (任意)
    const colorPalette = [
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
        '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080',
        '#FFFFFF', '#e2228b', '#22e28b', '#262e9b', '#888888'
    ];

    try {
        // [★変更点★] 推論サイズ(640×480) → 元画像サイズ(width×height) への拡大率
        const scaleX = width / inputSize.w;   // 幅方向
        const scaleY = height / inputSize.h;  // 高さ方向

        const strokeWidth = 2; // 枠線の太さ

        // <rect> 要素を生成 (スケーリング適用)
        const rects = batchno_classid_score_x1y1x2y2s.map(det => {
            const [batchNo, classId, score, x1, y1, x2, y2] = det;

            // 640×480 での座標 → 元サイズにスケーリング
            const rx1 = x1 * scaleX;
            const ry1 = y1 * scaleY;
            const rx2 = x2 * scaleX;
            const ry2 = y2 * scaleY;

            const w = rx2 - rx1;
            const h = ry2 - ry1;

            // クラスIDが 25 以上なら mod を取る (万一想定外クラスが検出された場合に対応)
            const strokeColor = colorPalette[classId % colorPalette.length];

            return `
                <rect
                    x="${rx1}"
                    y="${ry1}"
                    width="${w}"
                    height="${h}"
                    fill="none"
                    stroke="${strokeColor}"
                    stroke-width="${strokeWidth}"
                />
                <!-- (score やクラスIDなどを表示したい場合) -->
                <text
                    x="${rx1}"
                    y="${Math.max(ry1 - 5, 0)}"
                    fill="${strokeColor}"
                    font-size="16"
                    font-weight="bold"
                    stroke="#000"
                    stroke-width="0.5"
                    paint-order="stroke"
                >
                    class: ${classId}, score: ${score.toFixed(2)}
                </text>
            `;
        }).join('');

        // 全体の SVG を文字列で作成
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

        // Sharp で元画像(4チャネル, width×height)に SVG を合成
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

        // Debug用.png出力
        await sharp(imageWithBoxes, {
            raw: {
                channels: 4,
                width,
                height
            }
        })
        .png()
        .toFile(path.join(__dirname, 'debug_output.png'))
        .catch(err => {
            console.error('Failed to save debug_output.png:', err);
        });

        // 出力用に Uint8ClampedArray を返す
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
