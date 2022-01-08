// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


sqdmullt z0.h, z1.b, z2.b
// CHECK-INST: sqdmullt z0.h, z1.b, z2.b
// CHECK-ENCODING: [0x20,0x64,0x42,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 64 42 45 <unknown>

sqdmullt z29.s, z30.h, z31.h
// CHECK-INST: sqdmullt z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x67,0x9f,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: dd 67 9f 45 <unknown>

sqdmullt z31.d, z31.s, z31.s
// CHECK-INST: sqdmullt z31.d, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x67,0xdf,0x45]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: ff 67 df 45 <unknown>

sqdmullt z0.s, z1.h, z7.h[7]
// CHECK-INST: sqdmullt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xec,0xbf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 ec bf 44 <unknown>

sqdmullt z0.d, z1.s, z15.s[1]
// CHECK-INST: sqdmullt	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xec,0xef,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 ec ef 44 <unknown>
