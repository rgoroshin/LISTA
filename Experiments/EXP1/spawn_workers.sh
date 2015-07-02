#!/bin/bash

#spawn jobs 
ssh vine10 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 1 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine10 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 2 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine10 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 3 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine10 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 4 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine11 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 1 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine11 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 2 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine11 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 3 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine11 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 4 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine12 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 1 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine12 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 2 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine12 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 3 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine12 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 4 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine14 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 1 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine14 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 2 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine14 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 3 > ./Results/Experiments/output.txt" &
sleep 3 
ssh vine14 "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 4 > ./Results/Experiments/output.txt" &
sleep 3 
ssh juno "cd ~/Projects/LISTA; th ./Experiments/EXP1/worker.lua 1 > ./Results/Experiments/output.txt" &
