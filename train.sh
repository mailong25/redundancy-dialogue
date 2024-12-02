GPU_DEVICE=1
OPTIM="adafactor"
BATCH_SIZE=4
METHOD="unlikelihood"
CKPT="./checkpoints/$METHOD" 
VALID_STEPS=16100
TASKS="blended_skill_talk,msc:Session1Self:is_convai2_session_level=False,msc:SessionBaseMsc:session_id=2,msc:SessionBaseMsc:session_id=3,msc:SessionBaseMsc:session_id=4,msc:NegativeMsc:session_id=6,wizard_of_internet"

rm -rf ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2.py

if [[ $METHOD == "unlikelihood" ]]
then
  echo "unlikelihood training"
  cp -a ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2_ul.py ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2.py
elif [[ $METHOD == "unlikelihood_aug" ]]
then
  echo "unlikelihood_aug training"
  cp -a ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2_ul_aug.py ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2.py
else
  echo "contrastive training"
  cp -a ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2_cons.py ParlAI-1.6.0/projects/blenderbot2/agents/blenderbot2.py
fi

mkdir -p $CKPT
rm -rf $CKPT/*

trap 'kill $BGPID; exit' INT
python checkpoint_saving.py --path $CKPT &
BGPID=$!

export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

parlai train_model -dp ./ParlAI-1.6.0/data \
--model projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent \
--num_epochs 50 \
--query-model bert_from_parlai_rag --generation-model bart \
--rag-model-type token --rag-retriever-type search_engine --search_server None \
--dpr-model-file zoo:hallucination/bart_rag_token/model \
--gold-document-titles-key __select-docs-titles__ --insert-gold-docs True --model-parallel False \
--inference beam --learningrate 5e-06 \
--memory-key personas --memory-decoder-beam-min-length 3 \
--search-query-generator-model-file zoo:blenderbot2/query_generator/model --search-query-generator-beam-min-length 2 \
--knowledge-access-method memory_only \
--fp16 True --fp16-impl mem_efficient --force-fp16-tokens True --lr-scheduler-patience 100 \
--embedding-size 2560 --ffn-size 10240 --dropout 0.0 --attention-dropout 0.0 --n-heads 32 \
--learn-positional-embeddings False --embeddings-scale True --n-positions 128 --variant prelayernorm \
--activation relu --n-encoder-layers 2 --n-decoder-layers 24 \
--generation-model transformer/generator --beam-size 10 --beam-min-length 20 --beam-context-block-ngram -1 \
--beam-block-ngram 3 --history-add-global-end-token end --dict-tokenizer bytelevelbpe \
--dict-file ./ParlAI-1.6.0/data/models/blenderbot2/blenderbot2_3B/model.dict \
--bpe-vocab ./ParlAI-1.6.0/data/models/blenderbot2/blenderbot2_3B/model.dict-vocab.json \
--bpe-merge ./ParlAI-1.6.0/data/models/blenderbot2/blenderbot2_3B/model.dict-merges.txt \
--beam-block-full-context False --warmup-updates 100 --skip-generation True --checkpoint-activations True \
--truncate 128 --text-truncate 128 --label-truncate 128 --min-doc-token-length 64 --max-doc-token-length 64 \
--validation-metric ppl --n-docs 30 --n-ranked-doc-chunks 2 --splitted-chunk-length 64 --doc-chunks-ranker head \
--save-every-n-secs 480000 --validation_every_n_secs 720000 \
--evaltask blended_skill_talk,msc:Session1Self:is_convai2_session_level=False,msc:SessionBaseMsc:session_id=2,msc:SessionBaseMsc:session_id=3,msc:SessionBaseMsc:session_id=4,msc:NegativeMsc:session_id=6,wizard_of_internet \
--log-every-n-steps 180000 --update-freq 1 --save-after-valid True --datatype train:stream --log-every-n-secs 200 \
--memory-decoder-model-file '' --init-model ./ParlAI-1.6.0/data/models/blenderbot2/blenderbot2_3B/model \
--task $TASKS --validation_every_n_steps $VALID_STEPS --optimizer $OPTIM --batchsize $BATCH_SIZE \
--model-file $CKPT/model > $CKPT/log.txt

kill $BGPID