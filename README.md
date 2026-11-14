# audio-spectrum-character-code
音声スペクトラム・キャラクターコードの実験プロトタイプ

# 音声スペクトラム・キャラクターコード

## 目的

音声を文字列だけへ変換する前に、時間・周波数・強度を含むスペクトラム構造として保持する。確認済みの発音例を個人用の経験データベースへ蓄積し、発音内容・話者・感情表現を分けて認識する。

## 基本変数

音声波形を `x(t)` とし、短時間フーリエ変換を

```text
X(τ, f) = ∫ x(t) w(t-τ) exp(-i 2π f t) dt

とする。

AudioCharacterCode = {
  時間 τ,
  周波数 f,
  強度 |X(τ,f)|²,
  位相 arg(X(τ,f)),
  音の時間変化,
  ラベル（内容・話者・感情）
}

計算上は時間×周波数の行列であり、強度を高さまたは色として表示すると3Dチャートになる。

ここでいう「一つの変数」とは一個の数ではなく、音声全体をまとめた構造化されたデータオブジェクトを意味する。

3つの認識層

同じ音声パターンを一つのラベルへ混ぜず、次の三層を分ける。

1. 内容層：何を発音したか


2. 個人層：誰が発音したか


3. 表現層：どのように発音したか



内容層には、日本語の音、英語の文字・音素、単語などを記録する。

個人層には、声質、フォルマント、音域、発音癖などを記録する。

表現層には、怒り、喜び、悲しみ、平静、強弱、速度、間などを記録する。

この分離により、同じ「A」でも話者が異なる場合、また同じ話者でも感情が異なる場合を比較できる。

経験データベース

確認済みの音声例を、次のような記録として保存する。

{
  音声スペクトラムコード,
  発音内容,
  話者ID,
  感情ラベル,
  録音条件,
  確認状態,
  登録日時
}

AIが認識を誤った場合に、本人が正しいラベルを与えた例だけを確認済みデータとして保存する。

未確認の推測を自動的に学習させると、誤認識が固定化される可能性があるためである。

公開禁止

音声ファイルだけでなく、スペクトログラム、位相、音声特徴量、話者埋め込み、音声ハッシュ、SQLite経験データベースも公開しない。

これらは個人の声の識別や音声クローンに利用される可能性があるため、実験データはローカル環境に限定する。

公開対象は、理論・コード・再現手順だけとする。

音のつながり

単独の音だけでなく、隣接する音の時間変化を保持する。

単音コード A
  ＋
前後の音との遷移コード
  ＋
時間方向の強度・周波数変化
  ↓
音節・単語・文の認識

日本語の音の連続、英語の連結発音、個人特有の音の省略や変化を同じ枠組みで扱う候補になる。

音楽への拡張

音楽でも同じコード構造を使える。

内容ラベルを音符・楽器・和音などへ置き換え、表現層にテンポ、音量変化、音色、ビブラート、リズム、展開を記録する。

音楽スペクトラム
  ↓
音高・音色・リズム・強弱・時間変化
  ↓
音楽表現コード

ただし、音響パターンから推定できるのは「悲しげに聞こえる」などの表現傾向であり、演奏者の内面や意図を完全に確定するものではない。

通常コンピューターでの実証

量子コンピューターは必須ではない。通常のCPU・GPU・NPUで次の処理を実行できる。

WAV入力
  ↓
STFT
  ↓
スペクトラムコード抽出
  ↓
SQLite経験データベースへ保存
  ↓
類似度計算
  ↓
内容・話者・感情の推定

audio_character_code.py は、この前段階にあたるオフライン実験版である。

現版は、確認済み例を追加し、スペクトラムの類似度による最近傍認識を行う。

これは学習済み大規模モデルではなく、経験データベースの動作確認を目的とする。

実験手順

1. 音声例を登録する

python3 audio_character_code.py add sample_a.wav \
  --speaker jun --content あ --emotion calm

同じ話者で「あ」「い」「う」や、喜び・悲しみ・怒りの発音を複数回登録する。

2. 認識する

python3 audio_character_code.py recognize test.wav

内容・話者・感情ごとに、保存例との類似度上位を表示する。

類似度は確率ではなく、経験データベース上の近さである。

3. スペクトラムコードを保存する

python3 audio_character_code.py export sample_a.wav sample_a_code.npz

現版の限界

入力はPCM WAVであり、マイク録音機能は別途追加する。

現版の分類は最近傍比較であり、ニューラルネットワークではない。

主な認識特徴は正規化した強度スペクトラムである。

位相は保存するが、分類にはまだ全面的に使っていない。

音声だけから本当の感情を断定することはできない。

話者識別は確率的であり、雑音、体調、加齢、録音再生の影響を受ける。

未確認データの自動学習は行わない。


次の拡張

1. マイク入力とリアルタイムSTFT


2. 本人の訂正を受ける確認画面


3. 音素・文字列・単語遷移の学習


4. 話者・内容・感情を分離するニューラルモデル


5. 位相と非対角な時間・周波数関係を使ったQ3DOS型の比較


6. 音楽の音色・テンポ・感情表現への拡張



位置づけ

この構想は、量子コンピューターがなくても通常コンピューター上で検証を開始できる。

量子方式は、その後に高次元の相互作用計算を試す候補として位置づける。



## License

This project is licensed under the Apache License 2.0.
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including the
      original version of the Work and any modifications or additions to
      that Work or Derivative Works thereof, that is intentionally submitted
      to Licensor for inclusion in the Work by the copyright owner or by an
      individual or Legal Entity authorized to submit on behalf of the
      copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable by
      such Contributor that are necessarily infringed by their Contribution(s)
      alone or by combination of their Contribution(s) with the Work to which
      such Contribution(s) was submitted. If You institute patent litigation
      against any entity (including a cross-claim or counterclaim in a
      lawsuit) alleging that the Work or a Contribution incorporated within
      the Work constitutes direct or contributory patent infringement, then
      any patent licenses granted to You under this License for that Work
      shall terminate as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the Work
      or Derivative Works thereof in any medium, with or without modifications,
      and in Source or Object form, provided that You meet the following
      conditions:

      (a) You must give any other recipients of the Work or Derivative Works
          a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works that
          You distribute, all copyright, patent, trademark, and attribution
          notices from the Source form of the Work, excluding those notices
          that do not pertain to any part of the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one of
          the following places: within a NOTICE text file distributed as
          part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and do not
          modify the License. You may add Your own attribution notices
          within Derivative Works that You distribute, alongside or as an
          addendum to the NOTICE text from the Work, provided that such
          additional attribution notices cannot be construed as modifying
          the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing
      the origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or agreed
      to in writing, Licensor provides the Work (and each Contributor
      provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES
      OR CONDITIONS OF ANY KIND, either express or implied, including,
      without limitation, any warranties or conditions of TITLE,
      NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
      PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the Work
      (including but not limited to damages for loss of goodwill, work
      stoppage, computer failure or malfunction, or any and all other
      commercial damages or losses), even if such Contributor has been
      advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing the
      Work or Derivative Works thereof, You may choose to offer, and charge
      a fee for, acceptance of support, warranty, indemnity, or other
      liability obligations and/or rights consistent with this License.
      However, in accepting such obligations, You may act only on Your own
      behalf and on Your sole responsibility, not on behalf of any other
      Contributor, and only if You agree to indemnify, defend, and hold each
      Contributor harmless for any liability incurred by, or claims asserted
      against, such Contributor by reason of your accepting any such warranty
      or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!) The text should be enclosed in the appropriate
      comment syntax for the file format.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
