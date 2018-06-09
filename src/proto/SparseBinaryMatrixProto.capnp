@0xa55b6da98ffa08df;

# Next ID: 3
struct SparseBinaryMatrixProto {
  numRows @0 :UInt32;
  numColumns @1 :UInt32;
  indices @2 :List(List(UInt32));
}
