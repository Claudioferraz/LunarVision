# Como executar o LunarVision no Windows

## 1. Instalar o Python

Instale o Python 3.10 ou superior pelo site oficial. Durante a instalação, marque a opção **Add Python to PATH**.

## 2. Abrir o terminal na pasta do projeto

Abra a pasta `LunarVision`, clique na barra de endereço do Explorador de Arquivos, digite `cmd` e pressione Enter.

## 3. Instalar as bibliotecas

O LunarVision usa a API **MediaPipe Tasks**, compatível com as versões recentes do MediaPipe. Execute no mesmo terminal usado para rodar o programa:

```bat
python -m pip install --upgrade pip
python -m pip install --upgrade opencv-python numpy pyautogui Pillow mediapipe
```

Não use uma cópia antiga do arquivo que contenha `mp.solutions`. Se o VS Code continuar executando outra versão do arquivo, confirme que o caminho exibido no terminal é a pasta correta do projeto.

Na primeira execução, o programa baixará automaticamente o arquivo `hand_landmarker.task` para a mesma pasta de `lunarvision.py`. É necessário estar conectado à internet nessa primeira execução.

## 4. Iniciar o programa

Execute:

```bat
python lunarvision.py
```

## 5. Escolher e conectar a webcam

Ao abrir, o programa procura até dez índices de câmera usando dois métodos do Windows: **DirectShow** e **Media Foundation**. As opções aparecem no campo **Webcam**, por exemplo `Índice 0 — DirectShow` ou `Índice 1 — Media Foundation`. Se a CL-EyeTest estiver instalada, ela poderá aparecer em qualquer um desses índices e métodos. Selecione uma opção e pressione **Conectar** até aparecer a imagem correta. O botão **Atualizar** repete a busca.

Se nenhuma opção aparecer, confirme se o driver CL-EyeTest está instalado, se a câmera aparece no aplicativo Câmera do Windows e se nenhum outro programa está usando o dispositivo. Feche também aplicativos como Zoom, Teams, Discord ou OBS, pois eles podem bloquear a webcam. Depois reinicie o LunarVision e pressione **Atualizar**.

## 6. Calibrar

Pressione **Calibrar**. Na imagem exibida pelo programa, clique no canto superior esquerdo da área útil da lousa e depois no canto inferior direito. O retângulo azul mostrará a região reconhecida. Ao concluir, o programa ativará automaticamente o rastreamento do dedo indicador.

## 7. Testar o cursor e o clique

Mostre o **dedo indicador** para a webcam. O cursor deverá acompanhar a ponta do dedo dentro da área calibrada. Na imagem, o programa exibirá primeiro `Lousa: X ...% | Y ...%`; esses são os valores independentes da resolução. Depois, internamente, esses percentuais são convertidos para os pixels da resolução atual do Windows apenas para posicionar o cursor e clicar no slide. Para clicar, toque ou aponte para a posição desejada e mantenha o dedo praticamente parado por aproximadamente 0,35 segundo. O LunarVision interpretará essa permanência como um toque e executará `pyautogui.click()` uma única vez. Afaste ou mova o dedo antes do próximo toque para rearmar o clique.

> Durante os primeiros testes, mantenha o cursor sobre uma área segura e use o botão Encerrar para interromper o programa.

## Erros comuns do MediaPipe

Se aparecer `ModuleNotFoundError: No module named 'mediapipe'`, instale usando o mesmo interpretador que executará o programa:

```bat
python -m pip install --upgrade mediapipe
```

Se aparecer `module 'mediapipe' has no attribute 'solutions'`, o arquivo ainda está usando a API antiga. Substitua-o pela versão atualizada do projeto, que usa `mediapipe.tasks`. Se aparecer `HandLandmarkerOptions.__init__() got an unexpected keyword argument 'max_results'`, remova esse parâmetro: a versão atual usa `num_hands=1`.

## 8. Gestos disponíveis

O cursor sempre acompanha a ponta do dedo indicador. O que muda com o gesto é o comportamento do botão do mouse:

- **☝️ Só o indicador esticado** — apontar: move o cursor, sem clicar.
- **✌️ Indicador + médio esticados** — clique: dispara um clique único assim que o gesto é reconhecido (com um pequeno intervalo mínimo entre cliques, para não repetir por acidente).
- **✊ Mão fechada** — arraste: pressiona o botão esquerdo e mantém pressionado enquanto o punho continuar fechado; solta ao abrir a mão.
- **🤏 Polegar e indicador em pinça** — desenho: também mantém o botão pressionado (útil para o modo caneta de apresentações); solta ao afastar os dedos.
- **✋ Mão aberta (4 ou 5 dedos esticados)** — pausa: o cursor congela na posição atual, útil para reposicionar a mão sem mexer no cursor.

Na imagem do programa, o texto **"Gesto: ..."** mostra qual gesto está sendo reconhecido em tempo real.

## Limitações atuais

A primeira versão usa um retângulo definido por dois pontos e rastreia o dedo indicador com visão computacional. O toque é aproximado pela permanência do dedo quase parado; portanto, ainda não há sensor de profundidade para confirmar o contato físico real com a superfície. A correção de perspectiva e a integração com Arduino serão adicionadas posteriormente.
