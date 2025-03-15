let p1_hand = []; let p2_hand = []
let p1_point = 0; let p2_point = 0
let p1_selected_card = []; let p2_selected_card = []

const card_num = 8
let WIN_POINT = card_num*30+10
let WIN_TURN = 10

let dropped_cards_p1 = []; let dropped_cards_p2 = []

let turn = "p1"
let time = "game"
let numTurn = 1
let p1_is_acting = false

const elementToNumber = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,"Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,"Fe": 26, "Cu": 29, "Zn": 30, "I": 53}
const elements = [...Array(6).fill('H'), ...Array(4).fill('O'), ...Array(4).fill('C'),'He', 'Li', 'Be', 'B', 'N', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca','Fe', 'Cu', 'Zn', 'I']
const element = ['H','O','C','He', 'Li', 'Be', 'B', 'N', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca','Fe', 'Cu', 'Zn', 'I']
let deck = [...elements, ...elements]
let materials = []
let imageCache = {}

let model1;
let modelName1;
let model2;
let modelName2;


const countTemplate = Object.fromEntries(Object.values(elementToNumber).map(num => [num, 0]));


function convertToCount() {
    // テンプレートのコピーを作成
    let count = { ...countTemplate };
    // 配列内の各元素をカウント
    dropped_cards_p2.forEach(elem => {
        let num = elementToNumber[elem];
        if (num !== undefined) {
            count[num] += 1;
        }
    });
    // カウントの値を配列として返す（数値順に並ぶ）
    return Object.values(count);
}

let xs = [];
let ys = [];
let isTraining = false; // 学習中フラグ

function extractModelName(url) {
    const match = url.match(/\/([^\/]+)$/);
    return match ? match[1] : null;
}

// 1. モデルをロード（localStorageを優先）
async function loadModel() {
    try {
        const models = await tf.io.listModels();
        modelName1 = "model1";
        modelName2 = "model2";

        if (models['indexeddb://model1']) {
            model1 = await tf.loadLayersModel('indexeddb://model1'); // IndexedDB からロード
            console.log("✅ model1 を IndexedDB からロードしました");
        } else {
            model1 = await tf.loadLayersModel('https://kurorosuke.github.io/AI_models/model1/model.json'); // 外部URLからロード
            console.log("🌍 model1 をサーバーからロードしました");
            await model1.save('indexeddb://model1'); // ローカルに保存
        }

        if (models['indexeddb://model2']) {
            model2 = await tf.loadLayersModel('indexeddb://model2'); // IndexedDB からロード
            console.log("✅ model2 を IndexedDB からロードしました");
        } else {
            model2 = await tf.loadLayersModel('https://kurorosuke.github.io/AI_models/model1/model.json'); // 外部URLからロード
            console.log("🌍 model2 をサーバーからロードしました");
            await model2.save('indexeddb://model2'); // ローカルに保存
        }

        document.getElementById("Attention").style.display = "none";
    } catch (error) {
        console.error("🚨 モデルのロードに失敗しました", error);
        document.getElementById("Attention").style.display = "block";
    }
}


// 2. 追加データを学習用に変換
async function addTrainingData(playerData, generatedMaterialIndex, who) {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    // 入力データを取得
    var inputData = await convertToCount(playerData);
    var total = inputData.reduce(function(sum, element){return sum + element;}, 0);
    inputData.push(who);
    inputData.push(total*2 + Number(!who) + 1);
    console.log("学習用データ:", inputData);

    // データをTensorに変換
    const inputTensor = tf.tensor2d([inputData], [1, 26]);
    const outputTensor = tf.tensor2d([oneHotEncode(generatedMaterialIndex, materials.length)], [1, model.outputShape[1]]);
    console.log(outputTensor)

    // データセットに追加
    xs.push(inputTensor);
    ys.push(outputTensor);
    console.log("データを追加しました: クラス", generatedMaterialIndex);
}

// 3. モデルの追加学習
async function trainModel(model) {
    if (!model || xs.length === 0 || ys.length === 0) {
        console.warn("⚠️ 学習データが不足しています（xsまたはysが空）");
        console.log("📊 現在の xs のデータ数:", xs.length);
        console.log("📊 現在の ys のデータ数:", ys.length);
        return;
    }

    if (isTraining) {
        console.log("⚠️ すでに学習中...");
        return;
    }

    isTraining = true;
    console.log("🧠 モデルの追加学習を開始...");

    try {
        const xTrain = tf.concat(xs);
        const yTrain = tf.concat(ys);

        // **デバッグ情報を出力**
        console.log(`📊 学習データサイズ: xTrain=${xTrain.shape}, yTrain=${yTrain.shape}`);

        model.compile({
            optimizer: tf.train.adam(0.003),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        await model.fit(xTrain, yTrain, {
            epochs: 2,
            batchSize: 4,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`📈 Epoch ${epoch + 1}: Loss=${logs.loss.toFixed(4)}, MSE=${logs.mse.toFixed(4)}`);
                }
            }
        });

        console.log("✅ モデルの追加学習が完了しました");

        // **データを消さずに蓄積**
        xs = xs.slice(-500);
        ys = ys.slice(-500);
    } catch (error) {
        console.error("🚨 学習中にエラーが発生:", error);
    }

    isTraining = false;
    await model.save(`indexeddb://${model === model1 ? "model1" : "model2"}`);
}


async function runAutoBattle(numGames = 100) {
    for (let i = 0; i < numGames; i++) {
        console.log(`🔥 【ゲーム ${i + 1}】開始 🔥`);

        // **ゲームの初期化**
        resetGame();

        while (!checkWinCondition()) {
            await autoPlayTurn("p1"); // `model1` のターン
            if (checkWinCondition()) break;
            await autoPlayTurn("p2"); // `model2` のターン
        }

        // **報酬計算**
        let p1_reward = p1_point - p2_point;
        let p2_reward = p2_point - p1_point;

        await addTrainingDataForModel(model1, "p1", p1_reward);
        await addTrainingDataForModel(model2, "p2", p2_reward);

        // **モデルの学習 & 保存**
        await trainModel(model1);
        await trainModel(model2);

        console.log(`🎉 【ゲーム ${i + 1}】終了: P1(${p1_point}) vs P2(${p2_point})`);
    }
    console.log("✅ すべての自動対戦が終了しました！");
}


async function autoPlayTurn(player) {
    let model = player == "p1" ? model1 : model2;
    if (!model) {
        console.log(`${player} のモデルがロードされていません`);
        return;
    }

    // **最適な化合物を予測**
    let predictedMaterial = await runModelForPlayer(player);
    
    if (predictedMaterial) {
        // **役が作れる場合**
        await done(player);
    } else {
        // **役が作れない場合は、ランダムなカードを交換**
        let targetElem = Math.floor(Math.random() * 8);
        await exchangeCard(player, targetElem);
    }
}

async function runModelForPlayer(player) {
    let model = player === "p1" ? model1 : model2;
    if (!model) {
        console.log(`${player} のモデルがロードされていません`);
        return null;
    }

    // **ゲームの状態を数値化**
    let inputData = convertGameStateToInput(player);
    inputData = tf.tensor2d([inputData], [1, 26]);

    // **推論を実行**
    const output = model.predict(inputData);
    let outputData = await output.data();

    // **最もポイントが高い化合物を選択**
    let bestIndex = outputData.indexOf(Math.max(...outputData));
    let bestMaterial = materials[bestIndex] || null;

    console.log(`${player} の推論: ${bestMaterial ? bestMaterial.a : "なし"}`);
    return bestMaterial;
}

function checkWinCondition() {
    return Math.abs(p1_point - p2_point) >= WIN_POINT || numTurn >= WIN_TURN;
}


async function exchangeCard(player, targetElem) {
    let hand = player === "p1" ? p1_hand : p2_hand;
    let dropped_cards = player === "p1" ? dropped_cards_p1 : dropped_cards_p2;

    // **手札からカードを1枚捨てる**
    dropped_cards.push(hand[targetElem]);

    // **新しいカードをデッキから引く**
    let newElem = drawCard();
    hand[targetElem] = newElem;
}





function CanCreateMaterial(material) {
    // 必要な元素リスト
    const requiredElements = material.d;

    // 使用可能な元素のカウント
    const availableElements = {};

    // 使用可能なカードをすべて統合
    const allCards = [...deck, ...p2_hand];

    // 各カードの元素をカウント
    allCards.forEach(card => {
        availableElements[card] = (availableElements[card] || 0) + 1;
    });

    // 必要な元素が揃っているか確認
    for (const element in requiredElements) {
        if (!availableElements[element] || availableElements[element] < requiredElements[element]) {
            return true; // 必要な元素が不足している 「不足していなかったら」なのでここで反転させておく
        }
    }

    return false; // 全ての必要な元素が揃っている
}

function getUsedMaterials() {
    // localStorage から "materials" のデータを取得
    let storedMaterials = localStorage.getItem("materials");

    // データが null, 空文字, 空オブジェクトの場合は処理しない
    if (!storedMaterials || storedMaterials === "{}") {
        console.log("No valid materials data found.");
        return {};
    }

    // JSON をパース
    let materials = JSON.parse(storedMaterials);

    // 1回以上作成された（値が1以上の）物質のみを抽出
    let usedMaterials = Object.fromEntries(
        Object.entries(materials).filter(([key, value]) => value > 0)
    );

    return usedMaterials;
}

function calculatePseudoProbabilities(materials) {
    let total = Object.values(materials).reduce((sum, value) => sum + value, 0);
    if (total === 0) return {}; // すべて 0 なら確率なし

    let probabilities = {};
    for (let key in materials) {
        probabilities[key] = materials[key] / total;
    }

    return probabilities;
}

function calculateWeightedProbabilities(probabilities, outputData) {
    let weightedProbabilities = {};

    // 共通するキーがあれば掛け算し * 100、なければ outputData*0.1 にする
    for (let key in outputData) {
        if (probabilities.hasOwnProperty(key)) {
            sumNs = new Int8Array(localStorage.getItem("sumNs"))
            weightedProbabilities[key] = (probabilities[key]*sumNs / (sumNs + 10) + outputData[key]) /2; //\frac{x}{x+c} という関数で0→0、∞→1となる関数。cで速さを調整可能。
        } else {
            weightedProbabilities[key] = outputData[key];
        }
    }

    return weightedProbabilities;
}


async function addTrainingDataForModel(model, player, reward) {
    let inputData = convertGameStateToInput(player);

    // **モデルの出力次元を取得**
    let numClasses = model.outputShape[1];

    // **入力データを Tensor に変換**
    let inputTensor = tf.tensor2d([inputData], [1, 26]);

    // **One-Hot エンコーディングを修正（numClassesに統一）**
    let outputArray = new Array(numClasses).fill(0);
    let actionIndex = Math.floor(Math.random() * numClasses);
    outputArray[actionIndex] = reward;

    let outputTensor = tf.tensor2d([outputArray], [1, numClasses]);

    // **デバッグログ**
    console.log(`📝 ${player} の学習データ追加:`, {
        inputShape: inputTensor.shape,
        outputShape: outputTensor.shape,
        reward
    });

    // **追加前に次元をチェック**
    if (xs.length > 0 && xs[0].shape[1] !== inputTensor.shape[1]) {
        console.error(`🚨 ${player} の入力データの形状が一致しません！`, xs[0].shape, inputTensor.shape);
    } else {
        xs.push(inputTensor);
    }

    if (ys.length > 0 && ys[0].shape[1] !== outputTensor.shape[1]) {
        console.error(`🚨 ${player} の出力データの形状が一致しません！`, ys[0].shape, outputTensor.shape);
    } else {
        ys.push(outputTensor);
    }
}




function convertGameStateToInput(player) {
    let hand = player === "p1" ? p1_hand : p2_hand;
    let dropped_cards = player === "p1" ? dropped_cards_p1 : dropped_cards_p2;
    let deck_size = deck.length; // 残りのデッキのサイズ

    let count = { ...countTemplate };

    // **手札のカウント**
    hand.forEach(elem => {
        let num = elementToNumber[elem];
        if (num !== undefined) count[num] += 1;
    });

    // **捨てられたカードのカウント**
    dropped_cards.forEach(elem => {
        let num = elementToNumber[elem];
        if (num !== undefined) count[num] += 0.5;
    });

    // **配列に変換**
    let inputArray = Object.values(count);

    // **自分が作った（1）か相手が作った（0）か**
    let generated_by = player === "p1" ? 1 : 0;

    // **ターン情報**
    let turn_info = turn === "p1" ? 1 : 0;

    // **デッキサイズ（相対値として正規化）**
    let deck_ratio = deck_size / 36; //36枚を最大値

    // **追加情報を含める**
    inputArray.push(generated_by); // 自分が作ったなら1、相手なら0
    inputArray.push(turn_info); // 今のターンのプレイヤー
    //inputArray.push(deck_ratio); // 残りのデッキ割合

    return inputArray;
}





async function runModelForPlayer(player) {
    let model = player === "p1" ? model1 : model2;
    if (!model) {
        console.log(`${player} のモデルがロードされていません`);
        return null;
    }

    // **ゲームの状態を数値化**
    let inputData = convertGameStateToInput(player);
    
    inputData = tf.tensor2d([inputData], [1, 26]);  // 26次元の特徴ベクトル

    // **推論を実行**
    const output = model.predict(inputData);
    let outputData = await output.data();

    // **最適な化合物を選択**
    let predictedClass = outputData.indexOf(Math.max(...outputData));

    console.log(`推論結果: ${player} -> クラス ${predictedClass}`);
    return materials[predictedClass] || null;
}



//推論
async function runModel(who) {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    // 入力データ
    var inputData = await convertToCount();
    var total = inputData.reduce(function(sum, element){return sum + element;}, 0);
    inputData.push(who);
    inputData.push(total*2 + Number(!who) +1);

    inputData = tf.tensor2d([inputData], [1, 26]);

    // 推論実行
    const output = model.predict(inputData);
    let outputData = await output.data();

    recordCreatedMaterials = getUsedMaterials()
    pseudoProbability = calculatePseudoProbabilities(recordCreatedMaterials)

    let weightedResults = calculateWeightedProbabilities(pseudoProbability, outputData);
    console.log(pseudoProbability)


    // Math.max を使って最大値を取得
    var confidence = Math.max(...Object.values(weightedResults));

    // 最大値に対応するキーを検索
    var predictedClass = Object.keys(weightedResults).find(key => weightedResults[key] === confidence);

    confidences = output
    while (await CanCreateMaterial(materials[predictedClass])) {
        // weightedResults から現在の predictedClass を削除
        delete weightedResults[predictedClass];
    
        if (Object.keys(weightedResults).length === 0) {
            console.log("作成できる候補がありません");
            return;
        }
    
        // Math.max を使って最大値を取得
        var confidence = Math.max(...Object.values(weightedResults));
    
        // 最大値に対応するキーを検索（数値型に変換）
        var predictedClass = Object.keys(weightedResults).find(key => weightedResults[key] === confidence);
    }
    

    // 結果を表示
    console.log(`推論結果: クラス ${predictedClass}, 信頼度: ${confidence}`);
    document.getElementById("predictResult").innerHTML = `予測結果：${materials[predictedClass].a}・信頼度：${confidence}`;


    return { predictedClass, confidence };
}


// 5. 学習済みモデルを IndexedDB に保存
async function saveModel() {
    if (!model) {
        console.log("モデルがロードされていません");
        return;
    }

    try {
        console.log(modelName)
        console.log(`indexeddb://${modelName}`)
        await model.save(`indexeddb://${modelName}`); // IndexedDB に保存
        console.log("学習済みモデルを IndexedDB に保存しました");
    } catch (error) {
        console.error("モデルの保存に失敗しました", error);
    }
}

// One-Hot エンコーディング関数
function oneHotEncode(index, numClasses) {
    const encoded = new Array(numClasses).fill(0);
    encoded[index] = 1;
    return encoded;
}

//　load materials
async function loadMaterials(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (!data.material || !Array.isArray(data.material)) {
            document.getElementById("Attention2").style.display = "inline";
            return [];
        }
        document.getElementById("Attention2").style.display = "none";
        return data.material;
    } catch (error) {
        console.error("Error fetching compounds:", error);  // Log the error to the console for debugging
        document.getElementById("Attention2").style.display = "inline";
        return []; // Return an empty array in case of error
    }
}



// main code
async function view_p2_hand() {
    const area = document.getElementById('p2_hand')
    p2_hand.forEach((elem, index) => {
        const image = document.createElement("img")
        image.src = imageCache[elementToNumber[elem]].src
        image.alt = elem
        image.style.padding = "5px"
        image.style.border = "1px solid #000"
        image.classList.add("selected")
        image.addEventListener("click", function() {
            const button = document.getElementById("ron_button")
            button.style.display = "none"
            if (time == "make") {
                this.classList.toggle("selected")
                if (this.classList.contains("selected")){
                    this.style.border = "1px solid #000"
                    this.style.padding = "5px"
                    p2_selected_card.splice(p2_selected_card.indexOf(this.alt),1)
                } else {
                    this.style.border = "5px solid #F00"
                    this.style.padding = "1px"
                    p2_selected_card.push(this.alt)
                }}
            if (turn == "p2" && time == "game") {
                dropped_cards_p2.push(this.alt)
                const img = document.createElement("img")
                img.alt = this.alt
                img.src = imageCache[elementToNumber[this.alt]].src
                img.style.border = "1px solid #000"
                document.getElementById("dropped_area_p2").appendChild(img)
                this.classList.remove("selected")
                this.classList.add("selected")
                let newElem = drawCard()
                this.src = imageCache[elementToNumber[newElem]].src
                this.alt = newElem
                this.style.padding = "5px"
                this.style.border = "1px solid #000"
                p2_hand[index] = newElem
                turn = "p1"
                setTimeout(() => {p1_action()},500)
            }
        })
        area.appendChild(image)
    })
}

async function view_p1_hand() {
    const area = document.getElementById('p1_hand')
    p1_hand.forEach((elem, index) => {
        const image = document.createElement("img")
        image.src = imageCache[0].src
        image.alt = "相手の手札"
        image.style.padding = "5px"
        image.style.border = "1px solid #000"
        image.classList.add("selected")
        area.appendChild(image)
    })
}

async function search(components) {
    return materials.find(material => {
        for (const element in components) {
            if (!material.d[element] || material.d[element] !== components[element]) {
                return false;
            }
        }
        for (const element in material.d) {
            if (!components[element]) {
                return false;
            }
        }
        return true;
    }) || materials[0];
}

async function p1_make(predictedMaterialP2) {
    const makeable_material = await search_materials(arrayToObj(p1_hand));

    // 作れる物質がない場合は "なし" を返す
    if (!makeable_material || makeable_material.length === 0) {
        return [{
            "a": "なし",
            "b": "なし",
            "c": 0,
            "d": {},
            "e": []
        }];
    }

    // ポイントが高い順にソート
    makeable_material.sort((a, b) => b.c - a.c);

    return makeable_material;
}

async function p2_make() {
    // ボタンの表示を変更
    document.getElementById("generate_button").style.display = "none";
    const button = document.getElementById("done_button");
    button.style.display = "inline";

    // 以前のイベントリスナーを削除
    button.replaceWith(button.cloneNode(true));
    const newButton = document.getElementById("done_button");

    // ボタンクリックを待機
    return new Promise((resolve) => {
        newButton.addEventListener("click", function () {
            const p2_make_material = search(arrayToObj(p2_selected_card));
            resolve(p2_make_material);
        });
    });
}


async function get_dora() {
    return element[Math.round(Math.random()*23)]
}

async function incrementMaterialCount(material) {
    // localStorage から "materials" キーのデータを取得
    let materialsData = localStorage.getItem("materials");

    // JSONをパース（データがない場合は空のオブジェクトを設定）
    let materials = materialsData ? JSON.parse(materialsData) : {};

    // 指定された material の値を1増やす（存在しない場合は初期値1）
    materials[material] = (materials[material] || 0) + 1;

    // 更新したオブジェクトをJSONに変換してlocalStorageに保存
    localStorage.setItem("materials", JSON.stringify(materials));
    var sumNs = new Int8Array(localStorage.getItem("sumNs"))
    localStorage.setItem("sumNs", (sumNs)+1)
}


async function done() {
    // P1（model1）の推論 & 化合物作成
    const p1_make_material = await runModelForPlayer("p1");
    // P2（model2）の推論 & 化合物作成
    const p2_make_material = await runModelForPlayer("p2");

    let p1_point_gain = p1_make_material ? p1_make_material.c : -100;
    let p2_point_gain = p2_make_material ? p2_make_material.c : -100;

    // スコアを更新
    p1_point += p1_point_gain;
    p2_point += p2_point_gain;

    // **報酬の計算**
    let p1_reward = p1_point_gain - p2_point_gain;
    let p2_reward = p2_point_gain - p1_point_gain;

    // 学習データに追加
    await addTrainingDataForModel(model1, "p1", p1_reward);
    await addTrainingDataForModel(model2, "p2", p2_reward);

    // モデルの学習
    await trainModel(model1);
    await trainModel(model2);
}


async function win_check() {
    return Math.abs(p1_point - p2_point) >= WIN_POINT ? p1_point>p2_point ? "p1": "p2" : numTurn >= WIN_TURN ? p1_point>p2_point ? "p1": "p2" : null
}

async function p1_exchange(targetElem) {
    // Select a random card index from p1_hand// TODO: from AI.js
    dropped_cards_p1.push(p1_hand[targetElem])
    var exchange_element = p1_hand[targetElem]
    // Ensure the target card exists and is valid
    if (!p1_hand[targetElem]) {
        console.error("Invalid target element in p1_hand.")
        return
    }
    // Create a new image for the dropped card area
    const newImg = document.createElement("img")
    newImg.src = imageCache[elementToNumber[p1_hand[targetElem]]].src
    newImg.style.border = "1px solid #000"
    document.getElementById("dropped_area_p1").appendChild(newImg)
    // Update the player's hand with a new element
    const img = document.querySelectorAll("#p1_hand img")[targetElem]
    if (!img) {
        console.error("Image element not found in p1_hand.")
        return
    }
    // Select a new random element and replace the target card
    const newElem = drawCard()
    p1_hand[targetElem] = newElem
    // Update the image element's appearance
    img.src = imageCache[0].src
    img.alt = newElem
    img.style.padding = "5px"
    img.style.border = "1px solid #000"
    // Remove and reapply the 'selected' class to reset the state
    img.classList.remove("selected")
    img.classList.add("selected")
    // Switch the turn to "p1"
    turn = "p2"
    checkRon(exchange_element);
}

async function p1_action() {
    if (turn !== "p1" || p1_is_acting) {
        return;  // すでに行動中なら何もしない
    }
    p1_is_acting = true;  // 行動開始
    
    // フィルタリング
    const highPointMaterials = materials.filter(material => material.c > 20);
    
    // 最適な物質を選択
    const sortedMaterials = highPointMaterials.sort((a, b) => {
        let aMatchCount = Object.keys(a.d).reduce((count, elem) => count + Math.min(p1_hand.filter(e => e === elem).length, a.d[elem]), 0);
        let bMatchCount = Object.keys(b.d).reduce((count, elem) => count + Math.min(p1_hand.filter(e => e === elem).length, b.d[elem]), 0);
        return bMatchCount - aMatchCount || b.c - a.c;
    });

    const targetMaterial = sortedMaterials[0];

    if (!targetMaterial) {
        p1_exchange(Math.floor(Math.random() * p1_hand.length));
    } else {
        let canMake = true;
        for (const element in targetMaterial.d) {
            if (!p1_hand.includes(element) || p1_hand.filter(e => e === element).length < targetMaterial.d[element]) {
                canMake = false;
                break;
            }
        }

        if (canMake && targetMaterial.c > 20) {
            time = "make";
            await done("p1");
        } else {
            let unnecessaryCards = p1_hand.filter(e => {
                return !(e in targetMaterial.d) || p1_hand.filter(card => card === e).length > targetMaterial.d[e];
            });

            if (unnecessaryCards.length > 0) {
                let cardToExchange = unnecessaryCards[Math.floor(Math.random() * unnecessaryCards.length)];
                p1_exchange(p1_hand.indexOf(cardToExchange));
            } else {
                time = "make"
                done("p1");
            }
        }
    }
    
    turn = "p2";
    p1_is_acting = false;
}



//便利系関数
function arrayToObj(array) {
    let result = {}
    array.forEach(item => {
        if (result[item]) {
            result[item]++
        } else {
            result[item] = 1
        }
    })
    return result
}

function shuffle(array) {
    let currentIndex = array.length;
  
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
  
      // Pick a remaining element...
      let randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    };

    return array;
}

function drawCard() {
    return deck.length > 0 ? deck.pop() : (time = "make", done("no-draw"));
}

async function search_materials(components) {
    return materials.filter(material => {
        for (const element in material.d) {
            if (!components[element] || material.d[element] > components[element]) {
                return false;
            };
        };
        return true;
    });
}

function random_hand() {
    for (let i = 0; i < card_num; i++) {
        p1_hand.push(drawCard());
        p2_hand.push(drawCard());
    };
}

document.getElementById("generate_button").addEventListener("click", function () {
    if (turn == "p2") {
        time = "make"
        const newRonButton = document.getElementById("ron_button");
        newRonButton.style.display = "none";
        done("p2");
    }
})

function resetGame() {
    p1_hand = [];
    p2_hand = [];
    dropped_cards_p1 = [];
    dropped_cards_p2 = [];
    p1_selected_card = [];
    p2_selected_card = [];
    time = "game";
    turn = Math.random() <= 0.5 ? "p1" : "p2";
    numTurn = 1;  // ターンカウントをリセット

    document.getElementById("p1_point").innerHTML = `ポイント：${p1_point}`;
    document.getElementById("p1_explain").innerHTML = "　";
    document.getElementById("p2_point").innerHTML = `ポイント：${p2_point}`;
    document.getElementById("p2_explain").innerHTML = "　";
    document.getElementById("predictResult").innerHTML = "　";

    document.getElementById("generate_button").style.display = "inline";
    document.getElementById("done_button").style.display = "none";
    document.getElementById("nextButton").style.display = "none";

    deck = [...elements, ...elements];
    deck = shuffle(deck);

    const p1_hand_element = document.getElementById("p1_hand");
    const p2_hand_element = document.getElementById("p2_hand");
    p1_hand_element.innerHTML = "";
    p2_hand_element.innerHTML = "";

    const dropped_area_p1_element = document.getElementById("dropped_area_p1");
    const dropped_area_p2_element = document.getElementById("dropped_area_p2");
    dropped_area_p1_element.innerHTML = "";
    dropped_area_p2_element.innerHTML = "";

    random_hand();
    view_p1_hand();
    view_p2_hand();

    if (turn === "p1") {
        setTimeout(() => p1_action(), 500);
    }
}

function preloadImages() {
    let imageNumbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 26, 29, 30, 53];

    imageNumbers.forEach(num => {
        let img = new Image();
        img.src = `../images/${num}.webp`;
        imageCache[num] = img;
    });
}

async function init_json() {
    materials = await loadMaterials("https://kurorosuke.github.io/compounds/standard.json");
}



async function checkRon(droppedCard) {
    // P2のロン判定
    const possibleMaterialsP2 = await search_materials(arrayToObj([...p2_hand, droppedCard]));
    const validMaterialsP2 = possibleMaterialsP2.filter(material => material.d[droppedCard]);
    if (validMaterialsP2.length > 0) {
        const ronButton = document.getElementById("ron_button");
        ronButton.style.display = "inline";
        ronButton.replaceWith(ronButton.cloneNode(true));
        const newRonButton = document.getElementById("ron_button");

        newRonButton.addEventListener("click", function () {
            newRonButton.style.display = "none";
            const dropped = document.querySelectorAll("#dropped_area_p1 img");
            const selectCard = dropped[dropped.length - 1];
            selectCard.style.border = "2px solid red";
            selectCard.style.padding = "1px";
            p2_selected_card = [droppedCard];
            time = "make";
            // 捨て牌一覧の最後の要素を取得し、赤枠を付ける
            const DroppedCards = document.getElementById("dropped_area_p1").children
            const lastDiscard = DroppedCards[DroppedCards.length - 1]
            lastDiscard.style.border = "2px solid f00";
            done("p2", true);
        });
    }

    // P1のロン判定（捨てられたカードを含める）
    const possibleMaterialsP1 = await search_materials(arrayToObj([...p1_hand, droppedCard]));
    const validMaterialsP1 = possibleMaterialsP1.filter(material => ((material.c >= 70) && material.d[droppedCard]));

    if (validMaterialsP1.length > 0) {
        // **P1の手札に捨てたカードがもうない可能性があるため、戻す**
        p1_hand.push(droppedCard);
        // P1のロン処理のため、ロンに使うカードを選択
        p1_selected_card = [droppedCard];
        // `time` を "make" に変更
        time = "make";
        // P1のロン処理を実行
        done("p1", true);
    }
}

function updateGeneratedMaterials(materialName) {
    if (!materialName || materialName === "なし") return;

    // LocalStorage からデータを取得（なければ空のオブジェクト）
    let generatedMaterials = JSON.parse(localStorage.getItem("generatedMaterials")) || {};

    // 物質のカウントを更新
    if (generatedMaterials[materialName]) {
        generatedMaterials[materialName] += 1;
    } else {
        generatedMaterials[materialName] = 1;
    }

    // LocalStorage に保存
    localStorage.setItem("generatedMaterials", JSON.stringify(generatedMaterials));
}

//設定画面
function openWinSettings() {
    document.getElementById("winSettingsModal").style.display = "block";
}
async function saveWinSettings() {
    let winPointInput = parseInt(document.getElementById("winPointInput").value, 10);
    let winTurnInput = parseInt(document.getElementById("winTurnInput").value, 10);

    if (isNaN(winPointInput) || winPointInput < 1) {
        alert("WIN_POINT は 1 以上の数値を入力してください。");
        return;
    }
    if (isNaN(winPointInput) || winPointInput > 999) {
        alert("WIN_POINT の最大値は 999 です。");
        return;
    }
    if (isNaN(winTurnInput) || winTurnInput < 1) {
        alert("WIN_TURN は 1 以上の数値を入力してください。");
        return;
    }

    let compoundsValue = document.getElementById("compoundsSelection").value;
    if (compoundsValue != "url") {
        var compoundsURL = `https://kurorosuke.github.io/compounds/${compoundsValue}.json`;
    } else {
        var compoundsURL = document.getElementById("compoundsURL").value;
    }
    materials = await loadMaterials(compoundsURL);
    
    var modelSelect = document.getElementById("modelSelection").value
    if (modelSelect!="new"){
        modelURL = `https://kurorosuke.github.io/AI_models/${modelSelect}`
    } else {
        modelURL = document.getElementById("modelURL").value
    }
    model = loadModel()

    WIN_POINT = winPointInput;
    WIN_TURN = winTurnInput;
    closeWinSettings();
}
function closeWinSettings() {
    document.getElementById("winSettingsModal").style.display = "none";
}
document.getElementById("setting_icon").addEventListener("click", function() {
    document.getElementById("winSettingsModal").style.display = "inline"
})


//ヒント
async function findMostPointMaterial() {
    const possibleMaterials = await search_materials(arrayToObj(p2_hand));
    
    if (possibleMaterials.length === 0) {
        console.log("p2_hand 内で作成可能な物質はありません。");
    } else {
        const highestMaterial = possibleMaterials.reduce((max, material) => 
            material.c > max.c ? material : max, possibleMaterials[0]);
        console.log(`p2_hand 内で最もポイントが高い物質: ${highestMaterial.a} (ポイント: ${highestMaterial.c})`);
    }
}

function initializeMaterials() {
    // localStorage に "materials" が存在しない場合
    if (!localStorage.getItem("materials")) {
        // materials 内の各オブジェクトの a キーの値をキーとし、値を 0 にするオブジェクトを作成
        let initialMaterials = {};
        materials.forEach(item => {
            initialMaterials[item.a] = 0;
        });

        // 作成したオブジェクトを localStorage に保存
        localStorage.setItem("materials", JSON.stringify(initialMaterials));
    }
    if (!localStorage.getItem("sumNs")) {
        localStorage.setItem("sumNs", 0);
    }
}

document.addEventListener('DOMContentLoaded', function () {
    preloadImages()
    init_json()
    loadModel()
    initializeMaterials()
    deck = [...elements, ...elements]
    deck = shuffle(deck)
    random_hand()
    view_p1_hand()
    view_p2_hand()
    addOptions()
    turn = Math.random()>=0.5 ? "p1" : "p2"
    if (turn == "p1") {p1_action()}
})

function returnToStartScreen() {
    document.getElementById("startScreen").style.display = "flex";
    document.getElementById("p1_area").style.display = "none";
    document.getElementById("dropped_area_p1").style.display = "none";
    document.getElementById("dropped_area_p2").style.display = "none";
    document.getElementById("p2_area").style.display = "none";
    document.getElementById("gameRuleButton").style.display = "block";
}
document.getElementById("startButton").addEventListener("click", function() {
    document.getElementById("startScreen").style.display = "none";
    document.getElementById("p1_area").style.display = "block";
    document.getElementById("dropped_area_p1").style.display = "block";
    document.getElementById("dropped_area_p2").style.display = "block";
    document.getElementById("p2_area").style.display = "block";
    document.getElementById("gameRuleButton").style.display = "none";
});


function showRules() {
    document.getElementById("rulesModal").style.display = "block";
}

function closeRules() {
    document.getElementById("rulesModal").style.display = "none";
}

document.getElementById("closeRulesButton").addEventListener("click", closeRules);

// モーダル外をクリックした場合に閉じる
window.onclick = function(event) {
    const modal = document.getElementById("rulesModal");
    if (event.target === modal) {
        closeRules();
    }
};

function showInputTag() {
    if (document.getElementById("compoundsSelection").value == "url"){
        document.getElementById("compoundsURL").style.display = "inline";
    } else {
        document.getElementById("compoundsURL").style.display = "none";
    }
}

function showModelInputTag() {
    if (document.getElementById("modelSelection").value == "new"){
        document.getElementById("modelURL").style.display = "inline";
    } else {
        document.getElementById("modelURL").style.display = "none";
    }
}

async function getModelNames() {
    try {
        const models = await tf.io.listModels();
        const modelNames = Object.keys(models).map(key => key.replace('indexeddb://', ''));
        console.log(modelNames);
        return modelNames;
    } catch (error) {
        console.error("モデル名の取得に失敗しました", error);
        return [];
    }
}

async function addOptions() {
    let models = await getModelNames();
    models = models.slice(models.indexOf("model1")+1)
    const Selection = document.getElementById("modelSelection")
    models.forEach(elem => {
        const newOption = document.createElement("option");
        newOption.value = elem;
        newOption.text  = elem;
        Selection.appendChild(newOption)
    })
}
async function downloadModel(modelName) {
    try {
        // IndexedDB からモデルを取得
        const model = await tf.loadLayersModel(`indexeddb://${modelName}`);

        // モデルを `localStorage` にエクスポート
        await model.save(`downloads://${modelName}`);

        console.log(`✅ モデル ${modelName} をダウンロードフォルダに保存しました！`);
    } catch (error) {
        console.error(`🚨 モデル ${modelName} のダウンロードに失敗しました`, error);
    }
}

// モデルをダウンロード
//downloadModel("model1");
//downloadModel("model2");
