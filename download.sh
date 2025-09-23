#!/bin/bash

start=20200101
end=20250901
interval=5min

symbols=("BTCUSDT" "ETHUSDT" "BNBUSDT" "SOLUSDT" "XRPUSDT" "DOGEUSDT" "ADAUSDT" "AVAXUSDT" "TONUSDT" "DOTUSDT" "TRXUSDT" "SHIBUSDT" "LINKUSDT" "MATICUSDT" "LTCUSDT" "BCHUSDT")

for symbol in "${symbols[@]}"; do
    out="data/${symbol}_${interval}_${start}_${end}.csv"
    echo "正在下载 $symbol ..."
    proxychains python tsfm/download.py \
      --symbol $symbol \
      --interval $interval \
      --start $start \
      --end $end \
      --threads 8 \
      --out $out
done

echo "✅ 全部下载完成"
