@0xe1974e9322cc37d6;

# TODO: Use absolute path
using import "/proto/ConnectionsProto.capnp".ConnectionsProto;
using import "/proto/RandomProto.capnp".RandomProto;

# Next ID: 34
struct MotorTMProto {

  columnCount @0 :UInt32;
  cellsPerColumn @1 :UInt32;
  activationThreshold @2 :UInt32;
  initialPermanence @3 :Float32;
  connectedPermanence @4 :Float32;
  minThreshold @5 :UInt32;
  sampleSize @6 :UInt32;
  maxSynapsesPerSegment @7 :Int64;
  random @8 :RandomProto;

  apicalConnectionsD1 @9 :ConnectionsProto;
  apicalConnectionsD2 @10 :ConnectionsProto;

  # Lists of indices
  voluntaryActiveCells @11 :List(UInt32);
  winnerCells @12 :List(UInt32);

  learningRate @13 :Float32;
  apicalGlobalDecay @14 :Float32;
  winnerSize @15 :UInt32;
  punishPredDec @16 :Float32;
  motorCount @17:UInt32;


  motorCells @18 :List(UInt32);
  activeMotorCells @19 :List(UInt32);
  depolarizedApicalCells @20 :List(UInt32);
  depolarizedBasalCells @21 :List(UInt32);
  prevApicalInputD1 @22 :List(UInt32);
  prevApicalInputD2 @23 :List(UInt32);

}
