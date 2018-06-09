@0xdb5b6fad29f39828;

# Link to nupic proto schema
using import "/proto/UnionTemporalPoolerProto.capnp".UnionTemporalPoolerProto;

# Next ID: 4
struct MyTemporalPoolerRegionProto {
  # From ETM-Region
  poolerType @0 :Text;
  learningMode @1 :Bool;
  inferenceMode @2 :Bool;

  inputWidth @3:UInt32;
  columnCount @4:UInt32;
  historyLength @5:UInt32;
  minHistory @6:UInt32;

  # ETM Instance serialization
  pooler @7 :UnionTemporalPoolerProto;

}
