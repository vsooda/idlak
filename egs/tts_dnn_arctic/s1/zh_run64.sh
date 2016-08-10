source cmd.sh
source path.sh
H=`pwd`  #exp home
n=4
thchs=/home/sooda/data/thchs30-openslr
srate=16000
FRAMESHIFT=0.005
featdir=/home/sooda/features
corpus_dir=/home/sooda/data/tts/labixx120
test_dir=/home/sooda/data/tts/test
#lang=data/lang
#dict=data/dict
expa=exp-align
train=data/full
lang=data/lang_phone
dict=data/dict_phone

exp=exp_dnn
durdir=durdata
lbldurdir=lbldurdata
expdurdir=$exp/tts_dnn_dur_3_delta_quin5
dnndir=$exp/tts_dnn_train_3_deltasc2_quin5
#config
#0 not run; 1 run; 2 run and exit
DATA_PREP_MARY=0
LANG_PREP_PHONE64=0
EXTRACT_FEAT=0
ALIGNMENT_PHONE=0
GENERATE_LABLE=0
GENERATE_STATE=0
CONVERT_FEATURE=0
TRAIN_DNN=0
PACKAGE_DNN=0
VOCODER_TEST=0
spk="lbx"
audio_dir=$corpus_dir/wav 
prompt_lab=prompt_labels
state_lab=states
lab=labels

echo "##### Step 0: data preparation #####"
if [ $DATA_PREP_MARY -gt 0 ]; then
    #rm -rf data/{train,dev,full}
    rm -rf data/*
    mkdir -p data/{train,dev}
    mkdir -p data/full

    dev_pat='hs_zh_arctic_lbx_00??3'
    dev_rgx='hs_zh_arctic_lbx_00..3'
    train_pat='hs_zh_arctic_lbx_?????'
    train_rgx='hs_zh_arctic_lbx_.....'

    makeid="xargs -i basename {} .wav"

    find $audio_dir -iname "$train_pat".wav | sort | $makeid | awk -v audiodir=$audio_dir '{line=$1" "audiodir"/"$1".wav"; print line}' >> data/train/wav.scp

    find $audio_dir -iname "$dev_pat".wav | sort | $makeid | awk -v audiodir=$audio_dir '{line=$1" "audiodir"/"$1".wav"; print line}' >> data/dev/wav.scp

    grep "$train_rgx" $corpus_dir/phones.txt | sort >> data/train/text
    grep "$dev_rgx" $corpus_dir/phones.txt | sort >> data/dev/text

    for x in train dev; do
        cat data/$x/wav.scp | awk -v spk=$spk '{print $1, spk}' >> data/$x/utt2spk
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    done
    cat data/train/utt2spk data/dev/utt2spk | sort -u  > data/full/utt2spk
    cat data/train/text data/dev/text | sort -u > data/full/text
    cat data/train/wav.scp data/dev/wav.scp | sort -u > data/full/wav.scp
    utils/utt2spk_to_spk2utt.pl data/full/utt2spk | sort -u > data/full/spk2utt
    if [ $DATA_PREP_MARY -eq 2 ]; then
        echo "exit in data prepare"
        exit
    fi
fi


echo "make ph64"
if [ $LANG_PREP_PHONE64 -gt 0 ]; then
  rm -rf $dict $lang data/local/lang_phone
  cd $H; mkdir -p $dict $lang && \
  cp $corpus_dir/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} $dict  && \
  cat $corpus_dir/dict/lexicon.txt | grep -v '<eps>' | sort -u > $dict/lexicon.txt  && \
  echo "<SIL> sil " >> $dict/lexicon.txt  || exit 1;
  utils/prepare_lang.sh --num-nonsil-states 5 --share-silence-phones true --position_dependent_phones false $dict "<SIL>" data/local/lang_phone $lang || exit 1;
fi

#############################################
###### Step 1: acoustic data generation #####
#############################################
echo "##### Step 1: acoustic data generation #####"

if [ $EXTRACT_FEAT -gt 0 ]; then

    for step in train dev; do
        rm -f data/$step/feats.scp
        # Generate f0 features
        steps/make_pitch.sh --pitch-config conf/pitch.conf data/$step exp/make_pitch/$step   $featdir;
        cp data/$step/pitch_feats.scp data/$step/feats.scp
        # Compute CMVN on pitch features, to estimate min_f0 (set as mean_f0 - 2*std_F0)
        steps/compute_cmvn_stats.sh data/$step exp/compute_cmvn_pitch/$step $featdir;
        # For bndap / mcep extraction to be successful, the frame-length must be adjusted
        # in relation to the minimum pitch frequency.
        # We therefore do something speaker specific using the mean / std deviation from
        # the pitch for each speaker.
        min_f0=`copy-feats scp:"awk -v spk=$spk '(\\$1 == spk){print}' data/$step/cmvn.scp |" ark,t:- \
        | awk '(NR == 2){n = \$NF; m = \$2 / n}(NR == 3){std = sqrt(\$2/n - m * m)}END{print m - 2*std}'`
        echo $min_f0
        # Rule of thumb recipe; probably try with other window sizes?
        bndapflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 4.6 * 1000.0 / f0 + 0.5}'`
        mcepflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
        f0flen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
        echo "using wsizes: $bndapflen $mcepflen"
        subset_data_dir.sh --spk $spk data/$step 100000 data/${step}_$spk
        #cp data/$step/pitch_feats.scp data/${step}_$spk/
        # Regenerate pitch with more appropriate window
        steps/make_pitch.sh --pitch-config conf/pitch.conf --frame_length $f0flen data/${step}_$spk exp/make_pitch/${step}_$spk  $featdir;
        # Generate Band Aperiodicity feature
        steps/make_bndap.sh --bndap-config conf/bndap.conf --frame_length $bndapflen data/${step}_$spk exp/make_bndap/${step}_$spk  $featdir
        # Generate Mel Cepstral features
        #steps/make_mcep.sh  --sample-frequency $srate --frame_length $mcepflen  data/${step}_$spk exp/make_mcep/${step}_$spk   $featdir	
        steps/make_mcep.sh --sample-frequency $srate data/${step}_$spk exp/make_mcep/${step}_$spk   $featdir	
        # Merge features
        cat data/${step}_*/bndap_feats.scp > data/$step/bndap_feats.scp
        cat data/${step}_*/mcep_feats.scp > data/$step/mcep_feats.scp
        # Have to set the length tolerance to 1, as mcep files are a bit longer than the others for some reason
        paste-feats --length-tolerance=1 scp:data/$step/pitch_feats.scp scp:data/$step/mcep_feats.scp scp:data/$step/bndap_feats.scp ark,scp:$featdir/${step}_cmp_feats.ark,data/$step/feats.scp
        # Compute CMVN on whole feature set
        steps/compute_cmvn_stats.sh data/$step exp/compute_cmvn/$step   data/$step
    done

    if [ $EXTRACT_FEAT -eq 2 ]; then
        echo "exit in extract feature"
        exit
    fi
fi


#######################################
## 3a: create kaldi forced alignment ##
#######################################

echo "##### Step 3: forced alignment #####"
utils/fix_data_dir.sh data/full
utils/validate_lang.pl $lang

if [ $ALIGNMENT_PHONE -gt 0 ]; then
    rm -rf exp exp-align
    for step in full; do
      steps/make_mfcc.sh data/$step exp/make_mfcc/$step $featdir
      steps/compute_cmvn_stats.sh data/$step exp/make_mfcc/$step $featdir
    done
    # Now running the normal kaldi recipe for forced alignment
    test=data/eval_mfcc
    steps/train_mono.sh --boost-silence 0.25 --nj 1 --cmd "$train_cmd" \
                  $train $lang $expa/mono
    steps/align_si.sh --boost-silence 0.25 --nj 1 --cmd "$train_cmd" \
                $train $lang $expa/mono $expa/mono_ali
    steps/train_deltas.sh  --boost-silence 0.25 --cmd "$train_cmd" \
                 500 5000 $train $lang $expa/mono_ali $expa/tri1

    steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
                $train $lang $expa/tri1 $expa/tri1_ali
    steps/train_deltas.sh --cmd "$train_cmd" \
                 500 5000 $train $lang $expa/tri1_ali $expa/tri2

    # Create alignments
    steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
        $train $lang $expa/tri2 $expa/tri2_ali_full

    steps/train_deltas.sh --cmd "$train_cmd" \
        --context-opts "--context-width=5 --central-position=2" \
        500 5000 $train $lang $expa/tri2_ali_full $expa/quin

    # Create alignments
    steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
      $train $lang $expa/quin $expa/quin_ali_full

    ali=$expa/quin_ali_full
    # Extract phone alignment
    ali-to-phones --per-frame $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:- \
      | utils/int2sym.pl -f 2- $lang/phones.txt > $ali/phones.txt

    ali-to-hmmstate $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:$ali/states.tra

    if [ $ALIGNMENT_PHONE -eq 2 ]; then
        echo "exit after phone alignment"
        exit
    fi
fi

if [ $GENERATE_LABLE -gt 0 ]; then
    mkdir -p $prompt_lab
    rm $prompt_lab/*
    cat $expa/quin_ali_full/phones.txt | awk -v frameshift=$FRAMESHIFT -v labeldir=$prompt_lab '
    {
        lasttime = 0;
        lasttoken="";
        currenttime=0;
    }
    {
        outfile = labeldir"/"$1".lab";
        for(i=2;i<=NF;i++) {
            currenttime = currenttime + frameshift;
            if (lasttoken != "" && lasttoken != $i) {
                print lasttoken, lasttime, currenttime >> outfile
                lasttime = currenttime
            }
            lasttoken = $i; 
        }
        print lasttoken, lasttime, currenttime >> outfile
    }'

    if [ $GENERATE_LABLE -eq 2 ]; then
        echo "exit after phone alignment"
        exit
    fi
fi

if [ $GENERATE_STATE -gt 0 ]; then
    mkdir -p $state_lab
    rm $state_lab/*
    cat $expa/quin_ali_full/states.tra | awk -v statedir=$state_lab '
    {
        laststate = -1;
        counter = 0;
    }
    {
        outfile = statedir"/"$1".sta";
        for(i=2;i<=NF;i++) {
            if (laststate != $i && laststate != -1) {
                printf "%d %d ", laststate, counter > outfile
                if ($i == "0") printf("\n") >> outfile
                counter = 0
            }
            counter = counter + 1
            laststate = $i
        }
        printf "%d %d ", laststate, counter > outfile

    }'

    # paste the phone alignment and state aliment
    mkdir -p $lab
    rm -rf $lab/*
    for nn in `find $prompt_lab/*.lab | sort -u | xargs -i basename {} .lab`; do
        statename=$state_lab/$nn.sta
        labelname=$prompt_lab/$nn.lab
        mergelab=$lab/$nn.lab
        paste $labelname $statename -d '|' | sed 's/\t/ /g' > $mergelab
    done

    if [ $GENERATE_STATE -eq 2 ]; then
        echo "exit after state alignment"
        exit
    fi
fi



if [ $CONVERT_FEATURE -gt 0 ]; then
    step=full
    cat $corpus_dir/ali \
    | awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) print a[i]; print "]"}' \
    | copy-feats ark:- ark,t,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp

    # HACKY
    # Generate features for duration modelling
    # we remove relative position within phone and state
    copy-feats ark:$featdir/in_feats_full.ark ark,t:- \
    | awk -v nstate=5 'BEGIN{oldkey = 0; oldstate = -1; for (s = 0; s < nstate; s++) asd[s] = 0}
    function print_phone(vkey, vasd, vpd) {
      for (s = 0; s < nstate; s++) {
        print vkey, s, vasd[s], vpd;
        vasd[s] = 0;
      }
    }
    (NF == 2){print}
    (NF > 2){
      n = NF;
      if ($NF == "]") n = NF - 1;
      state = $(n-4); sd = $(n-3); pd = $(n-1);
      for (i = n-4; i <= NF; i++) $i = "";
      len = length($0);
      if (n != NF) len = len -1;
      key = substr($0, 1, len - 5);
      if ((key != oldkey) && (oldkey != 0)) {
        print_phone(oldkey, asd, opd);
        oldstate = -1;
      }
      if (state != oldstate) {
        asd[state] += sd;
      }
      opd = pd;
      oldkey = key;
      oldstate = state;
      if (NF != n) {
        print_phone(key, asd, opd);
        oldstate = -1;
        oldkey = 0;
        print "]";
      }
    }' > $featdir/tmp_durfeats_full.ark

    duration_feats="ark:$featdir/tmp_durfeats_full.ark"
    nfeats=$(feat-to-dim "$duration_feats" -)

    # Input
    select-feats 0-$(( $nfeats - 3 )) "$duration_feats" ark,scp:$featdir/in_durfeats_full.ark,$featdir/in_durfeats_full.scp
    # Output: duration of phone and state are assumed to be the 2 last features
    select-feats $(( $nfeats - 2 ))-$(( $nfeats - 1 )) "$duration_feats" ark,scp:$featdir/out_durfeats_full.ark,$featdir/out_durfeats_full.scp

    # Split in train / dev
    for step in train dev; do
      dir=lbldata/$step
      mkdir -p $dir
      #cp data/$step/{utt2spk,spk2utt} $dir
      utils/filter_scp.pl data/$step/utt2spk $featdir/in_feats_full.scp > $dir/feats.scp
      cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
      utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
      steps/compute_cmvn_stats.sh $dir $dir $dir
    done

    # Same for duration
    for step in train dev; do
      dir=lbldurdata/$step
      mkdir -p $dir
      #cp data/$step/{utt2spk,spk2utt} $dir
      utils/filter_scp.pl data/$step/utt2spk $featdir/in_durfeats_full.scp > $dir/feats.scp
      cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
      utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
      steps/compute_cmvn_stats.sh $dir $dir $dir

      dir=durdata/$step
      mkdir -p $dir
      #cp data/$step/{utt2spk,spk2utt} $dir
      utils/filter_scp.pl data/$step/utt2spk $featdir/out_durfeats_full.scp > $dir/feats.scp
      cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
      utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
      steps/compute_cmvn_stats.sh $dir $dir $dir
    done

    acdir=data
    lbldir=lbldata

    #ensure consistency in lists
    #for dir in $lbldir $acdir; do
    for class in train dev; do
      cp $lbldir/$class/feats.scp $lbldir/$class/feats_full.scp
      cp $acdir/$class/feats.scp $acdir/$class/feats_full.scp
      cat $acdir/$class/feats_full.scp | awk -v lst=$lbldir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $acdir/$class/feats.scp
      cat $lbldir/$class/feats_full.scp | awk -v lst=$acdir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $lbldir/$class/feats.scp
    done
    if [ $CONVERT_FEATURE -eq 2 ]; then
        echo "exit after convert feature"
        exit
    fi
fi


##############################
## 4. Train DNN
##############################

if [ $TRAIN_DNN -gt 0 ]; then
    echo "##### Step 4: training DNNs #####"

    mkdir -p $exp

    # Very basic one for testing
    #mkdir -p $exp
    #dir=$exp/tts_dnn_train_3e
    #$cuda_cmd $dir/_train_nnet.log steps/train_nnet_basic.sh --config conf/3-layer-nn.conf --learn_rate 0.2 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --bunch_size 50 --mlpOption " " --hid-dim 300 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dir

    echo " ### Step 4a: duration model DNN ###"
    # A. Small one for duration modelling
    rm -rf $expdurdir
    $cuda_cmd $expdurdir/_train_nnet.log steps/train_nnet_basic.sh --delta_order 2 --config conf/3-layer-nn-splice5.conf --learn_rate 0.02 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --max_iters 50 --randomize true --cache_size 50000 --bunch_size 200 --mlpOption " " --hid-dim 100 $lbldurdir/train $lbldurdir/dev $durdir/train $durdir/dev $expdurdir

    # B. Larger DNN for acoustic features
    echo " ### Step 4b: acoustic model DNN ###"

    rm -rf $dnndir
    $cuda_cmd $dnndir/_train_nnet.log steps/train_nnet_basic.sh --delta_order 2 --config conf/3-layer-nn-splice5.conf --learn_rate 0.04 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --cache_size 50000 --bunch_size 200 --mlpOption " " --hid-dim 700 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dnndir

    if [ $TRAIN_DNN -eq 2 ]; then
        echo "exit after train dnn"
        exit
    fi
fi

##############################
## 5. Synthesis
##############################

if [ "$srate" == "16000" ]; then
  order=39
  alpha=0.42
  fftlen=1024
  bndap_order=21
elif [ "$srate" == "48000" ]; then
  order=60
  alpha=0.55
  fftlen=4096
  bndap_order=25
elif [ "$srate" == "44100" ]; then
  order=60
  alpha=0.53
  fftlen=4096
  bndap_order=25
fi

echo "##### Step 5: synthesis #####"

if [ $VOCODER_TEST -gt 0 ]; then
    # Original samples:
    echo "Synthesizing vocoded training samples"
    mkdir -p exp_dnn/orig2/cmp exp_dnn/orig2/wav
    copy-feats scp:data/dev/feats.scp ark,t:- | awk -v dir=exp_dnn/orig2/cmp/ '($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'
    for cmp in exp_dnn/orig2/cmp/*.cmp; do
      utils/mlsa_synthesis_63_mlpg.sh --voice_thresh 0.5 --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $order $cmp exp_dnn/orig2/wav/`basename $cmp .cmp`.wav
    done
fi

# Variant with mlpg: requires mean / variance from coefficients
copy-feats scp:data/train/feats.scp ark:- \
| add-deltas --delta-order=2 ark:- ark:- \
| compute-cmvn-stats --binary=false ark:- - \
| awk '
(NR==2){count=$NF; for (i=1; i < NF; i++) mean[i] = $i / count}
(NR==3){if ($NF == "]") NF -= 1; for (i=1; i < NF; i++) var[i] = $i / count - mean[i] * mean[i]; nv = NF-1}
END{for (i = 1; i <= nv; i++) print mean[i], var[i]}' \
> data/train/var_cmp.txt

echo "  ###  5b: labixx samples synthesis ###"
# Alice test set
mkdir -p data/eval

# Generate CEX features for test set.
for step in eval; do
  # Generate input feature for duration modelling
  cat $test_dir/ali.test \
  | awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) for (state = 0; state < 5; state++) print a[i], state; print "]"}' \
  | copy-feats ark:- ark,scp:$featdir/in_durfeats_$step.ark,$featdir/in_durfeats_$step.scp
done

# Duration based test set
for step in eval; do
  dir=lbldurdata/$step
  mkdir -p $dir
  cp $featdir/in_durfeats_$step.scp $dir/feats.scp
  cut -d ' ' -f 1 $dir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
  steps/compute_cmvn_stats.sh $dir $dir $dir
done

# Generate label with DNN-generated duration
echo "Synthesizing MLPG eval samples"
#  1. forward pass through duration DNN
utils/make_forward_fmllr.sh $expdurdir $lbldurdir/eval $expdurdir/tst_forward/ ""

testAlignDir=test_labels
mkdir -p $testAlignDir
rm -rf $testAlignDir/* 

#  2. make the duration consistent, generate labels with duration information added
for cmp in $expdurdir/tst_forward/cmp/*.cmp; do
  cat $cmp | awk -v outdir=$testAlignDir -v nstate=5 -v id=`basename $cmp .cmp` '
  BEGIN{
    print "\"" id ".lab\""; 
    outfile = outdir"/"id".lab"
    tstart = 0 
  }
  {
    pd += $2;
    sd[NR % nstate] = $1
  }
  (NR % nstate == 0){
    mpd = pd / nstate;
    smpd = 0;
    for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
    rmpd = int((smpd + mpd) / 2 + 0.5);
    # Normal phones
    if (int(sd[0] + 0.5) == 0) {
      for (i = 1; i <= 3; i++) {
        sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[3] <= 0) sd[3] = 1;
      for (i = 4; i <= nstate; i++) sd[i % nstate] = 0;
    }
    # Silence phone
    else {
      for (i = 1; i <= nstate; i++) {
        sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[0] <= 0) sd[0] = 1;
    }
    if (sd[1] <= 0) sd[1] = 1;
    smpd = 0;
    for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
    for (i = 1; i <= nstate; i++) {
      if (sd[i % nstate] > 0) {
        tend = tstart + sd[i % nstate] * 50000;
        print tstart, tend, int(NR / 5), i-1 >> outfile
        tstart = tend;
      }
    }
    pd = 0;
  }'
done

# 3. Turn them into DNN input labels (i.e. one sample per frame)
for step in eval; do
  python utils/make_fullctx_mlf_dnn.py data/$step/synth_lab.mlf data/$step/cex.ark data/$step/feat.ark
  copy-feats ark:data/$step/feat.ark ark,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp
done
for step in eval; do
  dir=lbldata/$step
  mkdir -p $dir
  cp $featdir/in_feats_$step.scp $dir/feats.scp
  cut -d ' ' -f 1 $dir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
  steps/compute_cmvn_stats.sh $dir $dir $dir
done
# 4. Forward pass through big DNN
utils/make_forward_fmllr.sh $dnndir $lbldir/eval $dnndir/tst_forward/ ""

# 5. Vocoding
# NB: these are the settings for 16k
mkdir -p $dnndir/tst_forward/wav_mlpg/; for cmp in $dnndir/tst_forward/cmp/*.cmp; do
  utils/mlsa_synthesis_63_mlpg.sh --voice_thresh 0.5 --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $order --delta_order 2 $cmp $dnndir/tst_forward/wav_mlpg/`basename $cmp .cmp`.wav data/train/var_cmp.txt
done

if [ $PACKAGE_DNN -gt 0 ]; then
    echo "#### Step 6: packaging DNN voice ####"

    utils/make_dnn_voice.sh --spk $spk --srate $srate --mcep_order $order --bndap_order $bndap_order --alpha $alpha --fftlen $fftlen

    echo "Voice packaged successfully. Portable models have been stored in ${spk}_mdl."
    echo "Synthesis can be performed using:
    echo \"This is a demo of D N N synthesis\" | utils/synthesis_voice.sh ${spk}_mdl <outdir>"
fi
