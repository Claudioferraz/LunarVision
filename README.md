# LunarVision

Protótipo que usa a webcam e a MediaPipe Tasks para rastrear a mão e controlar o cursor do Windows, com gestos específicos para apontar, clicar, arrastar e desenhar.

## Como executar

Veja o passo a passo completo em [`Como_executar_o_LunarVision_no_Windows.md`](Como_executar_o_LunarVision_no_Windows.md).

Resumo rápido:

```bat
python -m pip install -r requirements.txt
python lunarvision.py
```

Na primeira execução, o programa baixa automaticamente o arquivo `hand_landmarker.task` (não incluído neste repositório).

## Gestos

- ☝️ Só o indicador esticado — apontar (move o cursor)
- ✌️ Indicador + médio esticados — clique
- ✊ Mão fechada — arraste
- 🤏 Polegar + indicador em pinça — desenho
- ✋ Mão aberta — pausa (congela o cursor)

## Status do projeto

Protótipo inicial: calibração por dois pontos (retângulo), sem correção de perspectiva e sem integração com Arduino ainda. Veja [`LunarVision — Protótipo para Windows.md`](LunarVision___Prot%C3%B3tipo_para_Windows.md) para o roadmap.
