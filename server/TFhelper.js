const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js for Node.js をインポート
const fs = require('fs'); // ファイルシステム操作のモジュールをインポート
const path = require('path'); // パス操作のモジュールをインポート
const { glob } = require('glob'); // ファイルパターンのマッチングモジュールをインポート

/**
 * 画像ファイルをテンソルに変換する関数
 * @param {string} filePath 画像ファイルのパス
 * @param {Array<number>} size 画像のリサイズサイズ (例: [224, 224])
 * @returns {tf.Tensor} 画像テンソル
 */
function fileToTensor(filePath, size) {
  const rawimage = fs.readFileSync(filePath); // 画像ファイルを同期的に読み込む
  const imageTensor = tf.node.decodeImage(rawimage, 3); // 画像データをテンソルに変換 (3チャンネル)
  const resizedTensor = tf.image.resizeBilinear(imageTensor, size); // 画像を指定サイズにリサイズ
  const normalizedTensor = tf.cast(resizedTensor.div(tf.scalar(255)), dtype = 'float32'); // テンソルの型を float32 に変換し、0-1 に正規化
  return normalizedTensor; // 正規化された画像テンソルを返す
}

/**
 * フォルダ内の画像をテンソルに変換する関数
 * @param {string} dirPath フォルダのパス
 * @param {Array<number>} size 画像のリサイズサイズ (例: [224, 224])
 * @returns {Promise<Array>} [正規化された画像テンソル, ラベルテンソル, ラベル名配列] を含む Promise
 */
function folderToTensors(dirPath, size) {
  return new Promise((resolve, reject) => {
    const XS = []; // 画像テンソルを格納する配列
    const YS = []; // ラベル (数値) を格納する配列
    const dirs = []; // ラベル名を格納する配列

    console.log('Identifying Image List'); // 処理状況を出力
    glob(`${dirPath}/*/*.@(png|jpeg|jpg|bmp)`) // フォルダ内の画像ファイルを検索
      .then(files => {
        console.log(`${files.length} Files Found`); // 見つかったファイル数を出力
        console.log('Now converting to tensors'); // 処理状況を出力

        files.forEach((file) => {
          const dir = path.basename(path.dirname(file)); // ファイルの親ディレクトリ名 (ラベル名) を取得
          if (!dirs.includes(dir)) {
            dirs.push(dir); // 新しいラベル名なら配列に追加
          }
          const answer = dirs.indexOf(dir); // ラベル名を数値に変換
          const imageTensor = fileToTensor(file, size); // 画像ファイルをテンソルに変換
          YS.push(answer); // ラベル (数値) を配列に追加
          XS.push(imageTensor); // 画像テンソルを配列に追加
        });

        /**
         * 画像テンソルとラベル (数値) の配列をシャッフルする関数
         * @param {Array} array 画像テンソルの配列
         * @param {Array} array2 ラベル (数値) の配列
         */
        function shuffleCombo(array, array2) {
          let counter = array.length;
          console.assert(array.length === array2.length); // 配列の長さが同じであることを確認
          let temp, temp2;
          let index = 0;
          while (counter > 0) {
            index = (Math.random() * counter) | 0; // ランダムなインデックスを取得
            counter--;
            // 配列の要素をスワップ
            temp = array[counter];
            temp2 = array2[counter];
            array[counter] = array[index];
            array2[counter] = array2[index];
            array[index] = temp;
            array2[index] = temp2;
          }
        }
        shuffleCombo(XS, YS); // 画像テンソルとラベル (数値) の配列をシャッフル

        console.log('Stacking'); // 処理状況を出力
        const X = tf.stack(XS); // 画像テンソルをスタック
        const Y = tf.oneHot(YS, dirs.length); // ラベル (数値) を one-hot エンコーディング

        console.log('Images all converted to tensors:'); // 処理状況を出力
        console.log('X', X.shape); // 変換後の画像テンソルの形状を出力
        console.log('Y', Y.shape); // 変換後のラベルテンソルの形状を出力

        // X を 0-1 に正規化
        const XNORM = X.div(255);
        // 不要なテンソルを解放
        tf.dispose([XS, X]); 

        resolve([XNORM, Y, dirs]); // Promise を解決し、[正規化された画像テンソル, ラベルテンソル, ラベル名配列] を返す
      })
      .catch(error => {
        console.error('Failed to access files', error); // ファイルアクセスエラーを出力
        reject(); // Promise を拒否
        process.exit(1); // プロセスを終了
      });
  });
}

/**
 * 転移学習を用いてモデルを学習する関数
 * @param {string} folderPath 学習データのフォルダパス
 * @param {object} socket Socket.IO のソケットオブジェクト (オプション)
 * @returns {Promise<void>} 学習完了後に解決される Promise
 */
async function learnTransferModel(folderPath, socket = null) {
  console.log('Loading images - this may take a while...'); // 処理状況を出力
  if (socket) { socket.emit('log', 'loading images'); } // Socket.IO でログを送信 (オプション)
  const [X, Y, dirs] = await folderToTensors(folderPath, [224, 224]); // フォルダ内の画像をテンソルに変換

  console.log('Loading model'); // 処理状況を出力
  if (socket) { socket.emit('log', 'loading model'); } // Socket.IO でログを送信 (オプション)

  // MobileNet v2 特徴量抽出モデルを読み込む
  const featureModel = await tf.loadGraphModel('https://www.kaggle.com/models/google/mobilenet-v2/TfJs/140-224-feature-vector/3', { fromTFHub: true }); 

  // 転移学習モデルを定義
  const transferModel = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [1792],
        units: 64,
        activation: 'relu',
      }),
      tf.layers.dense({ units: dirs.length, activation: 'softmax' }),
    ],
  });

  console.log('Creating features from images - this may take a while...'); // 処理状況を出力
  if (socket) { socket.emit('log', 'creating features'); } // Socket.IO でログを送信 (オプション)

  const featureX = featureModel.predict(X); // 特徴量抽出モデルで特徴量を抽出
  console.log(`Features stack ${featureX.shape}`); // 特徴量テンソルの形状を出力
  if (socket) { socket.emit('log', `Features stack ${featureX.shape}`); } // Socket.IO でログを送信 (オプション)

  // 転移学習モデルをコンパイル
  transferModel.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // モデルの評価
  console.log(transferModel.evaluate(featureX, Y)[1].dataSync());

  // 転移学習モデルを学習
  const history = await transferModel.fit(featureX, Y, {
    epochs: 100, // 学習エポック数
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // if (false) {
        //   model.stopTraining = true
        // }
        console.log(`epoch:${epoch} acc:${logs.acc}`); // エポックごとの精度を出力
        if (socket) { socket.emit('updateProgress', epoch); } // Socket.IO で進捗状況を送信 (オプション)
      },
    },
    verbose: false, // 学習ログの表示を抑制
  });

  console.log('learned!'); // 学習完了を出力
  if(socket){
    socket.emit('log','learned'); // Socket.IO でログを送信 (オプション)
    socket.emit('learnCompleted'); // 学習完了イベントを送信 
  }
}

// モジュールをエクスポート
module.exports = {
  learnTransferModel: learnTransferModel,
};

// ローカルで学習を実行 (コメントアウト)
// learnTransferModel('c:/users/kosan/desktop/sotsuken/images');