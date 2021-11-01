//I
precision highp float;
#define GLSLIFY 1

uniform sampler2D buffer;
uniform sampler2D tex;
uniform vec2 dims;
uniform float time;

uniform vec4 rect0;
uniform vec4 rect1;
uniform vec4 rect2;
uniform vec4 rect3;
uniform vec4 rect4;
uniform vec4 rect5;
uniform vec4 rect6;
uniform vec4 rect7;
uniform vec4 rect8;
uniform vec4 rect9;
uniform vec4 rect10;
uniform vec4 rect11;
uniform vec4 rect12;
uniform vec4 rect13;
uniform vec4 rect14;
uniform vec4 rect15;

varying vec2 vUv;

//#pragma glslify: cellular2D = require(\'./cellular2D\')
// Cellular noise ("Worley noise") in 2D in GLSL.
// Copyright (c) Stefan Gustavson 2011-04-19. All rights reserved.
// This code is released under the conditions of the MIT license.
// See LICENSE file for details.
// https://github.com/stegu/webgl-noise

// Modulo 289 without a division (only multiplications)
vec2 mod289_7(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289_7(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

// Modulo 7 without a division
vec4 mod7(vec4 x) {
  return x - floor(x * (1.0 / 7.0)) * 7.0;
}

// Permutation polynomial: (34x^2 + x) mod 289
vec4 permute_7(vec4 x) {
  return mod289_7((34.0 * x + 1.0) * x);
}

// Cellular noise, returning F1 and F2 in a vec2.
// Speeded up by using 2x2 search window instead of 3x3,
// at the expense of some strong pattern artifacts.
// F2 is often wrong and has sharp discontinuities.
// If you need a smooth F2, use the slower 3x3 version.
// F1 is sometimes wrong, too, but OK for most purposes.
vec2 cellular2x2(vec2 P) {
#define K 0.142857142857 // 1/7
#define K2 0.0714285714285 // K/2
#define jitter 0.8 // jitter 1.0 makes F1 wrong more often
  vec2 Pi = mod289_7(floor(P));
  vec2 Pf = fract(P);
  vec4 Pfx = Pf.x + vec4(-0.5, -1.5, -0.5, -1.5);
  vec4 Pfy = Pf.y + vec4(-0.5, -0.5, -1.5, -1.5);
  vec4 p = permute_7(Pi.x + vec4(0.0, 1.0, 0.0, 1.0));
  p = permute_7(p + Pi.y + vec4(0.0, 0.0, 1.0, 1.0));
  vec4 ox = mod7(p)*K+K2;
  vec4 oy = mod7(floor(p*K))*K+K2;
  vec4 dx = Pfx + jitter*ox;
  vec4 dy = Pfy + jitter*oy;
  vec4 d = dx * dx + dy * dy; // d11, d12, d21 and d22, squared
  // Sort out the two smallest distances
#if 0
  // Cheat and pick only F1
  d.xy = min(d.xy, d.zw);
  d.x = min(d.x, d.y);
  return vec2(sqrt(d.x)); // F1 duplicated, F2 not computed
#else
  // Do it right and find both F1 and F2
  d.xy = (d.x < d.y) ? d.xy : d.yx; // Swap if smaller
  d.xz = (d.x < d.z) ? d.xz : d.zx;
  d.xw = (d.x < d.w) ? d.xw : d.wx;
  d.y = min(d.y, d.z);
  d.y = min(d.y, d.w);
  return sqrt(d.xy);
#endif
}

//#pragma glslify: cellular2x2x2 = require(\'./cellular2x2x2\')
//#pragma glslify: cellular3D = require(\'./cellular3D\')
//
// Fractional Brownian motion
// https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
//
float rand(float n) {
  return fract(sin(n) * 43758.5453123);
}

float noise(float p) {
  float fl = floor(p);
  float fc = fract(p);
  return mix(rand(fl), rand(fl + 1.0), fc);
}

float rand(vec2 n) { 
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p){
  vec2 ip = floor(p);
  vec2 u = fract(p);
  u = u*u*(3.0-2.0*u);

  float res = mix(mix(rand(ip), rand(ip + vec2(1.0, 0.0)), u.x), mix(rand(ip + vec2(0.0, 1.0)), rand(ip + vec2(1.0, 1.0)), u.x), u.y);
  return res*res;
}

float mod289_8(float x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 mod289_8(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 perm(vec4 x) {
  return mod289_8(((x * 34.0) + 1.0) * x);
}

float noise(vec3 p) {
  vec3 a = floor(p);
  vec3 d = p - a;
  d = d * d * (3.0 - 2.0 * d);

  vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
  vec4 k1 = perm(b.xyxy);
  vec4 k2 = perm(k1.xyxy + b.zzww);

  vec4 c = k2 + a.zzzz;
  vec4 k3 = perm(c);
  vec4 k4 = perm(c + 1.0);

  vec4 o1 = fract(k3 * (1.0 / 41.0));
  vec4 o2 = fract(k4 * (1.0 / 41.0));

  vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
  vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

  return o4.y * d.y + o4.x * (1.0 - d.y);
}

#define NUM_OCTAVES 8

float fbm(float x) {
  float v = 0.0;
  float a = 0.5;
  float shift = float(100);
  for (int i = 0; i < NUM_OCTAVES; ++i) {
    v += a * noise(x);
    x = x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

float fbm(vec2 x) {
  float v = 0.0;
  float a = 0.5;
  vec2 shift = vec2(100);
  // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
  for (int i = 0; i < NUM_OCTAVES; ++i) {
    v += a * noise(x);
    x = rot * x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

float fbm(vec3 x) {
  float v = 0.0;
  float a = 0.5;
  vec3 shift = vec3(100);
  for (int i = 0; i < NUM_OCTAVES; ++i) {
    v += a * noise(x);
    x = x * 2.0 + shift;
    a *= 0.5;
  }
  return v;
}

//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289_4(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289_4(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute_4(vec3 x) {
  return mod289_4(((x*34.0)+1.0)*x);
}

float snoise_0(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289_4(i); // Avoid truncation effects in permutation
  vec3 p = permute_4( permute_4( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289_9(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289_9(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_8(vec4 x) {
     return mod289_9(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_6(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise_2(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g_0 = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g_0;
  vec3 i1 = min( g_0.xyz, l.zxy );
  vec3 i2 = max( g_0.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289_9(i);
  vec4 p = permute_8( permute_8( permute_8(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0_1 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt_6(vec4(dot(p0_1,p0_1), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0_1 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0_1,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

//
// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec4 mod289_5(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289_5(float x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute_5(vec4 x) {
     return mod289_5(((x*34.0)+1.0)*x);
}

float permute_5(float x) {
     return mod289_5(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_4(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float taylorInvSqrt_4(float r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
  {
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
  }

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451

float snoise_1(vec4 v)
  {
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

// First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

// Permutations
  i = mod289_5(i);
  float j0 = permute_5( permute_5( permute_5( permute_5(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute_5( permute_5( permute_5( permute_5 (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0_0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt_4(vec4(dot(p0_0,p0_0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0_0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt_4(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0_0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

  }

//
// GLSL textureless classic 2D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-08-22
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec4 mod289_2(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_2(vec4 x)
{
  return mod289_2(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_2(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec2 fade_2(vec2 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise_1(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289_2(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute_2(permute_2(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy_0 = abs(gx) - 0.5 ;
  vec4 tx_0 = floor(gx + 0.5);
  gx = gx - tx_0;

  vec2 g00 = vec2(gx.x,gy_0.x);
  vec2 g10 = vec2(gx.y,gy_0.y);
  vec2 g01 = vec2(gx.z,gy_0.z);
  vec2 g11 = vec2(gx.w,gy_0.w);

  vec4 norm = taylorInvSqrt_2(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade_2(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

//
// GLSL textureless classic 3D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-10-11
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec3 mod289_10(vec3 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289_10(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_9(vec4 x)
{
  return mod289_10(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_7(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade_5(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise_2(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289_10(Pi0);
  Pi1 = mod289_10(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute_9(permute_9(ix) + iy);
  vec4 ixy0 = permute_9(ixy + iz0);
  vec4 ixy1 = permute_9(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt_7(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt_7(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade_5(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

//
// GLSL textureless classic 4D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-08-22
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec4 mod289_1(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_1(vec4 x)
{
  return mod289_1(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_1(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 fade_1(vec4 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise_0(vec4 P)
{
  vec4 Pi0 = floor(P); // Integer part for indexing
  vec4 Pi1 = Pi0 + 1.0; // Integer part + 1
  Pi0 = mod289_1(Pi0);
  Pi1 = mod289_1(Pi1);
  vec4 Pf0 = fract(P); // Fractional part for interpolation
  vec4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = vec4(Pi0.zzzz);
  vec4 iz1 = vec4(Pi1.zzzz);
  vec4 iw0 = vec4(Pi0.wwww);
  vec4 iw1 = vec4(Pi1.wwww);

  vec4 ixy = permute_1(permute_1(ix) + iy);
  vec4 ixy0 = permute_1(ixy + iz0);
  vec4 ixy1 = permute_1(ixy + iz1);
  vec4 ixy00 = permute_1(ixy0 + iw0);
  vec4 ixy01 = permute_1(ixy0 + iw1);
  vec4 ixy10 = permute_1(ixy1 + iw0);
  vec4 ixy11 = permute_1(ixy1 + iw1);

  vec4 gx00 = ixy00 * (1.0 / 7.0);
  vec4 gy00 = floor(gx00) * (1.0 / 7.0);
  vec4 gz00 = floor(gy00) * (1.0 / 6.0);
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  vec4 gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  vec4 sw00 = step(gw00, vec4(0.0));
  gx00 -= sw00 * (step(0.0, gx00) - 0.5);
  gy00 -= sw00 * (step(0.0, gy00) - 0.5);

  vec4 gx01 = ixy01 * (1.0 / 7.0);
  vec4 gy01 = floor(gx01) * (1.0 / 7.0);
  vec4 gz01 = floor(gy01) * (1.0 / 6.0);
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  vec4 gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  vec4 sw01 = step(gw01, vec4(0.0));
  gx01 -= sw01 * (step(0.0, gx01) - 0.5);
  gy01 -= sw01 * (step(0.0, gy01) - 0.5);

  vec4 gx10 = ixy10 * (1.0 / 7.0);
  vec4 gy10 = floor(gx10) * (1.0 / 7.0);
  vec4 gz10 = floor(gy10) * (1.0 / 6.0);
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  vec4 gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  vec4 sw10 = step(gw10, vec4(0.0));
  gx10 -= sw10 * (step(0.0, gx10) - 0.5);
  gy10 -= sw10 * (step(0.0, gy10) - 0.5);

  vec4 gx11 = ixy11 * (1.0 / 7.0);
  vec4 gy11 = floor(gx11) * (1.0 / 7.0);
  vec4 gz11 = floor(gy11) * (1.0 / 6.0);
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  vec4 gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  vec4 sw11 = step(gw11, vec4(0.0));
  gx11 -= sw11 * (step(0.0, gx11) - 0.5);
  gy11 -= sw11 * (step(0.0, gy11) - 0.5);

  vec4 g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
  vec4 g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
  vec4 g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
  vec4 g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
  vec4 g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
  vec4 g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
  vec4 g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
  vec4 g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
  vec4 g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
  vec4 g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
  vec4 g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
  vec4 g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
  vec4 g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
  vec4 g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
  vec4 g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
  vec4 g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

  vec4 norm00 = taylorInvSqrt_1(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  vec4 norm01 = taylorInvSqrt_1(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  vec4 norm10 = taylorInvSqrt_1(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  vec4 norm11 = taylorInvSqrt_1(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  float n0000 = dot(g0000, Pf0);
  float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
  float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
  float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
  float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
  float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
  float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
  float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
  float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
  float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
  float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
  float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
  float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
  float n1111 = dot(g1111, Pf1);

  vec4 fade_xyzw = fade_1(Pf0);
  vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
  vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
  vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}

//
// GLSL textureless classic 2D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-08-22
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec4 mod289_3(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_3(vec4 x)
{
  return mod289_3(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_3(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec2 fade_3(vec2 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise, periodic variant
float pnoise_1(vec2 P, vec2 rep)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, rep.xyxy); // To create noise with explicit period
  Pi = mod289_3(Pi);        // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute_3(permute_3(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy_1 = abs(gx) - 0.5 ;
  vec4 tx_1 = floor(gx + 0.5);
  gx = gx - tx_1;

  vec2 g00 = vec2(gx.x,gy_1.x);
  vec2 g10 = vec2(gx.y,gy_1.y);
  vec2 g01 = vec2(gx.z,gy_1.z);
  vec2 g11 = vec2(gx.w,gy_1.w);

  vec4 norm = taylorInvSqrt_3(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade_3(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

//
// GLSL textureless classic 3D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-10-11
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec3 mod289_6(vec3 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289_6(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_6(vec4 x)
{
  return mod289_6(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_5(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade_4(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise, periodic variant
float pnoise_2(vec3 P, vec3 rep)
{
  vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
  vec3 Pi1 = mod(Pi0 + vec3(1.0), rep); // Integer part + 1, mod period
  Pi0 = mod289_6(Pi0);
  Pi1 = mod289_6(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute_6(permute_6(ix) + iy);
  vec4 ixy0 = permute_6(ixy + iz0);
  vec4 ixy1 = permute_6(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt_5(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt_5(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade_4(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

//
// GLSL textureless classic 4D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-08-22
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/ashima/webgl-noise
//

vec4 mod289_0(vec4 x)
{
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute_0(vec4 x)
{
  return mod289_0(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt_0(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 fade_0(vec4 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise, periodic version
float pnoise_0(vec4 P, vec4 rep)
{
  vec4 Pi0 = mod(floor(P), rep); // Integer part modulo rep
  vec4 Pi1 = mod(Pi0 + 1.0, rep); // Integer part + 1 mod rep
  Pi0 = mod289_0(Pi0);
  Pi1 = mod289_0(Pi1);
  vec4 Pf0 = fract(P); // Fractional part for interpolation
  vec4 Pf1 = Pf0 - 1.0; // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = vec4(Pi0.zzzz);
  vec4 iz1 = vec4(Pi1.zzzz);
  vec4 iw0 = vec4(Pi0.wwww);
  vec4 iw1 = vec4(Pi1.wwww);

  vec4 ixy = permute_0(permute_0(ix) + iy);
  vec4 ixy0 = permute_0(ixy + iz0);
  vec4 ixy1 = permute_0(ixy + iz1);
  vec4 ixy00 = permute_0(ixy0 + iw0);
  vec4 ixy01 = permute_0(ixy0 + iw1);
  vec4 ixy10 = permute_0(ixy1 + iw0);
  vec4 ixy11 = permute_0(ixy1 + iw1);

  vec4 gx00 = ixy00 * (1.0 / 7.0);
  vec4 gy00 = floor(gx00) * (1.0 / 7.0);
  vec4 gz00 = floor(gy00) * (1.0 / 6.0);
  gx00 = fract(gx00) - 0.5;
  gy00 = fract(gy00) - 0.5;
  gz00 = fract(gz00) - 0.5;
  vec4 gw00 = vec4(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
  vec4 sw00 = step(gw00, vec4(0.0));
  gx00 -= sw00 * (step(0.0, gx00) - 0.5);
  gy00 -= sw00 * (step(0.0, gy00) - 0.5);

  vec4 gx01 = ixy01 * (1.0 / 7.0);
  vec4 gy01 = floor(gx01) * (1.0 / 7.0);
  vec4 gz01 = floor(gy01) * (1.0 / 6.0);
  gx01 = fract(gx01) - 0.5;
  gy01 = fract(gy01) - 0.5;
  gz01 = fract(gz01) - 0.5;
  vec4 gw01 = vec4(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
  vec4 sw01 = step(gw01, vec4(0.0));
  gx01 -= sw01 * (step(0.0, gx01) - 0.5);
  gy01 -= sw01 * (step(0.0, gy01) - 0.5);

  vec4 gx10 = ixy10 * (1.0 / 7.0);
  vec4 gy10 = floor(gx10) * (1.0 / 7.0);
  vec4 gz10 = floor(gy10) * (1.0 / 6.0);
  gx10 = fract(gx10) - 0.5;
  gy10 = fract(gy10) - 0.5;
  gz10 = fract(gz10) - 0.5;
  vec4 gw10 = vec4(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
  vec4 sw10 = step(gw10, vec4(0.0));
  gx10 -= sw10 * (step(0.0, gx10) - 0.5);
  gy10 -= sw10 * (step(0.0, gy10) - 0.5);

  vec4 gx11 = ixy11 * (1.0 / 7.0);
  vec4 gy11 = floor(gx11) * (1.0 / 7.0);
  vec4 gz11 = floor(gy11) * (1.0 / 6.0);
  gx11 = fract(gx11) - 0.5;
  gy11 = fract(gy11) - 0.5;
  gz11 = fract(gz11) - 0.5;
  vec4 gw11 = vec4(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
  vec4 sw11 = step(gw11, vec4(0.0));
  gx11 -= sw11 * (step(0.0, gx11) - 0.5);
  gy11 -= sw11 * (step(0.0, gy11) - 0.5);

  vec4 g0000 = vec4(gx00.x,gy00.x,gz00.x,gw00.x);
  vec4 g1000 = vec4(gx00.y,gy00.y,gz00.y,gw00.y);
  vec4 g0100 = vec4(gx00.z,gy00.z,gz00.z,gw00.z);
  vec4 g1100 = vec4(gx00.w,gy00.w,gz00.w,gw00.w);
  vec4 g0010 = vec4(gx10.x,gy10.x,gz10.x,gw10.x);
  vec4 g1010 = vec4(gx10.y,gy10.y,gz10.y,gw10.y);
  vec4 g0110 = vec4(gx10.z,gy10.z,gz10.z,gw10.z);
  vec4 g1110 = vec4(gx10.w,gy10.w,gz10.w,gw10.w);
  vec4 g0001 = vec4(gx01.x,gy01.x,gz01.x,gw01.x);
  vec4 g1001 = vec4(gx01.y,gy01.y,gz01.y,gw01.y);
  vec4 g0101 = vec4(gx01.z,gy01.z,gz01.z,gw01.z);
  vec4 g1101 = vec4(gx01.w,gy01.w,gz01.w,gw01.w);
  vec4 g0011 = vec4(gx11.x,gy11.x,gz11.x,gw11.x);
  vec4 g1011 = vec4(gx11.y,gy11.y,gz11.y,gw11.y);
  vec4 g0111 = vec4(gx11.z,gy11.z,gz11.z,gw11.z);
  vec4 g1111 = vec4(gx11.w,gy11.w,gz11.w,gw11.w);

  vec4 norm00 = taylorInvSqrt_0(vec4(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
  g0000 *= norm00.x;
  g0100 *= norm00.y;
  g1000 *= norm00.z;
  g1100 *= norm00.w;

  vec4 norm01 = taylorInvSqrt_0(vec4(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
  g0001 *= norm01.x;
  g0101 *= norm01.y;
  g1001 *= norm01.z;
  g1101 *= norm01.w;

  vec4 norm10 = taylorInvSqrt_0(vec4(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
  g0010 *= norm10.x;
  g0110 *= norm10.y;
  g1010 *= norm10.z;
  g1110 *= norm10.w;

  vec4 norm11 = taylorInvSqrt_0(vec4(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
  g0011 *= norm11.x;
  g0111 *= norm11.y;
  g1011 *= norm11.z;
  g1111 *= norm11.w;

  float n0000 = dot(g0000, Pf0);
  float n1000 = dot(g1000, vec4(Pf1.x, Pf0.yzw));
  float n0100 = dot(g0100, vec4(Pf0.x, Pf1.y, Pf0.zw));
  float n1100 = dot(g1100, vec4(Pf1.xy, Pf0.zw));
  float n0010 = dot(g0010, vec4(Pf0.xy, Pf1.z, Pf0.w));
  float n1010 = dot(g1010, vec4(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
  float n0110 = dot(g0110, vec4(Pf0.x, Pf1.yz, Pf0.w));
  float n1110 = dot(g1110, vec4(Pf1.xyz, Pf0.w));
  float n0001 = dot(g0001, vec4(Pf0.xyz, Pf1.w));
  float n1001 = dot(g1001, vec4(Pf1.x, Pf0.yz, Pf1.w));
  float n0101 = dot(g0101, vec4(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
  float n1101 = dot(g1101, vec4(Pf1.xy, Pf0.z, Pf1.w));
  float n0011 = dot(g0011, vec4(Pf0.xy, Pf1.zw));
  float n1011 = dot(g1011, vec4(Pf1.x, Pf0.y, Pf1.zw));
  float n0111 = dot(g0111, vec4(Pf0.x, Pf1.yzw));
  float n1111 = dot(g1111, Pf1);

  vec4 fade_xyzw = fade_0(Pf0);
  vec4 n_0w = mix(vec4(n0000, n1000, n0100, n1100), vec4(n0001, n1001, n0101, n1101), fade_xyzw.w);
  vec4 n_1w = mix(vec4(n0010, n1010, n0110, n1110), vec4(n0011, n1011, n0111, n1111), fade_xyzw.w);
  vec4 n_zw = mix(n_0w, n_1w, fade_xyzw.z);
  vec2 n_yzw = mix(n_zw.xy, n_zw.zw, fade_xyzw.y);
  float n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
  return 2.2 * n_xyzw;
}

float map(float value, float inMin, float inMax, float outMin, float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 map(vec2 value, vec2 inMin, vec2 inMax, vec2 outMin, vec2 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec3 map(vec3 value, vec3 inMin, vec3 inMax, vec3 outMin, vec3 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec4 map(vec4 value, vec4 inMin, vec4 inMax, vec4 outMin, vec4 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

highp float random(vec2 co)
{
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

// return 1 if v inside the box, return 0 otherwise
float insideBox(vec2 v, vec2 bottomLeft, vec2 topRight) {
  vec2 s = step(bottomLeft, v) - step(topRight, v);
  return s.x * s.y;   
}

float insideBox3D(vec3 v, vec3 bottomLeft, vec3 topRight) {
  vec3 s = step(bottomLeft, v) - step(topRight, v);
  return s.x * s.y * s.z; 
}

void main() {
  vec3 rgb = texture2D(buffer, vUv).rgb;

  // rect0 (simplex & classic noise)
  if (insideBox(vUv, vec2(rect0.x / dims.x, rect0.y / dims.y), vec2((rect0.x + rect0.z) / dims.x, (rect0.y + rect0.w) / dims.y)) >= 1.0) {
    vec2 offset = vec2(snoise_2(vec3(vUv * 22.0, time)), cnoise_2(vec3(vUv * 7.0, time * 0.5))) / dims;
    rgb = texture2D(buffer, vUv + offset).rgb;
  }

  // rect1 (perlin noise)
  if (insideBox(vUv, vec2(rect1.x / dims.x, rect1.y / dims.y), vec2((rect1.x + rect1.z) / dims.x, (rect1.y + rect1.w) / dims.y)) >= 1.0) {
    float x = pnoise_2(vec3(vUv * 90.0, time * 1.2), vec3(0.34, 5.731, 0.12)) / dims.x;
    float y = pnoise_2(vec3(vUv.y * 53.84, time * 1.41, vUv.x * 20.14), vec3(0.89, 0.23, 0.911)) / dims.y;
    vec2 offset = vec2(x, y);
    rgb = texture2D(buffer, vUv + offset).rgb;
  }

  // rect2 (fbm noise)
  if (insideBox(vUv, vec2(rect2.x / dims.x, rect2.y / dims.y), vec2((rect2.x + rect2.z) / dims.x, (rect2.y + rect2.w) / dims.y)) >= 1.0) {
    float x = (fbm(vec3(vUv.x * 2.0, vUv.y, time * 5.0)) - 0.5) * 4.0 / dims.x * 1.0; // 2.0
    float y = (fbm(vec3(time * 3.0, vUv.x, vUv.y * 4.0)) - 0.5) * 8.0 / dims.y * 1.5; // 3.0
    vec2 offset = vec2(x, y);
    rgb = texture2D(buffer, vUv + offset).rgb;
  }

  // rect3 (fbm & simplex noise)
  if (insideBox(vUv, vec2(rect3.x / dims.x, rect3.y / dims.y), vec2((rect3.x + rect3.z) / dims.x, (rect3.y + rect3.w) / dims.y)) >= 1.0) {
    float x = (fbm(vec2((vUv.x - 0.5) * 2.0, time * 5.0)) - 0.35) * 4.0 / dims.x + snoise_2(vec3(vUv.x * 4.3, vUv.y, time * 0.4)) * 0.005; // 0.01
    float y = (fbm(vec2(time * 3.0, vUv.y * 4.0)) - 0.35) * 4.0 / dims.y + snoise_2(vec3(vUv.x * 3.3, vUv.y * 3.4, time * 0.38)) * 0.005; // 0.01
    vec2 offset = vec2(x, y);
    rgb = texture2D(buffer, vUv + offset).rgb;
  }

  // rect4 (x+)
  if (insideBox(vUv, vec2(rect4.x / dims.x, rect4.y / dims.y), vec2((rect4.x + rect4.z) / dims.x, (rect4.y + rect4.w) / dims.y)) >= 1.0) {
    float offset = 1.0 / dims.x;
    rgb = texture2D(buffer, vec2(vUv.x + offset, vUv.y)).rgb;
  }

  // rect5 (x-)
  if (insideBox(vUv, vec2(rect5.x / dims.x, rect5.y / dims.y), vec2((rect5.x + rect5.z) / dims.x, (rect5.y + rect5.w) / dims.y)) >= 1.0) {
    float offset = 1.0 / dims.x;
    rgb = texture2D(buffer, vec2(vUv.x - offset, vUv.y)).rgb;
  }

  // rect6 (y+)
  if (insideBox(vUv, vec2(rect6.x / dims.x, rect6.y / dims.y), vec2((rect6.x + rect6.z) / dims.x, (rect6.y + rect6.w) / dims.y)) >= 1.0) {
    float offset = 1.0 / dims.y;
    rgb = texture2D(buffer, vec2(vUv.x, vUv.y + offset)).rgb;
  }

  // rect7 (y-)
  if (insideBox(vUv, vec2(rect7.x / dims.x, rect7.y / dims.y), vec2((rect7.x + rect7.z) / dims.x, (rect7.y + rect7.w) / dims.y)) >= 1.0) {
    float offset = 1.0 / dims.y;
    rgb = texture2D(buffer, vec2(vUv.x, vUv.y - offset)).rgb;
  }

  // rect8 (animate)
  if (insideBox(vUv, vec2(rect8.x / dims.x, rect8.y / dims.y), vec2((rect8.x + rect8.z) / dims.x, (rect8.y + rect8.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect8.x;
    float y = floor(vUv.y * dims.y) - rect8.y;
    float step = mod(floor(time * 24.0), 16.0);

    if (step == 0.0 && mod(y, 2.0) == 0.0) {
      vec2 pos = vec2(vUv.x + 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 0.0 && mod(y, 2.0) == 1.0) {
      vec2 pos = vec2(vUv.x - 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 8.0 && mod(y, 2.0) == 0.0) {
      vec2 pos = vec2(vUv.x - 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 8.0 && mod(y, 2.0) == 1.0) {
      vec2 pos = vec2(vUv.x + 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }
  }

  // rect9 (animate)
  if (insideBox(vUv, vec2(rect9.x / dims.x, rect9.y / dims.y), vec2((rect9.x + rect9.z) / dims.x, (rect9.y + rect9.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect9.x;
    float y = floor(vUv.y * dims.y) - rect9.y;
    float step = mod(floor(time * 60.0), 4.0);

    if (step == 0.0 && mod(x, 2.0) == 0.0 && mod(y, 2.0) == 0.0) {
      vec2 pos = vec2(vUv.x + 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 0.0 && mod(x, 2.0) == 1.0 && mod(y, 2.0) == 0.0) {
      vec2 pos = vec2(vUv.x, vUv.y - 1.0 / dims.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 0.0 && mod(x, 2.0) == 1.0 && mod(y, 2.0) == 1.0) {
      vec2 pos = vec2(vUv.x - 1.0 / dims.x, vUv.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 0.0 && mod(x, 2.0) == 0.0 && mod(y, 2.0) == 1.0) {
      vec2 pos = vec2(vUv.x, vUv.y + 1.0 / dims.y);
      rgb = texture2D(buffer, pos).rgb;
    }
  }

  // rect10 (animate)
  if (insideBox(vUv, vec2(rect10.x / dims.x, rect10.y / dims.y), vec2((rect10.x + rect10.z) / dims.x, (rect10.y + rect10.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect10.x;
    float y = floor(vUv.y * dims.y) - rect10.y;
    float step = mod(floor(time * 60.0), 4.0);

    if (step == 0.0 && mod(x + y, 2.0) == 0.0) {
      vec2 pos = vec2(vUv.x + 1.0 / dims.x, vUv.y - 1.0 / dims.y);
      rgb = texture2D(buffer, pos).rgb;
    }

    if (step == 0.0 && mod(x + y, 2.0) == 1.0) {
      vec2 pos = vec2(vUv.x - 1.0 / dims.x, vUv.y + 1.0 / dims.y);
      rgb = texture2D(buffer, pos).rgb;
    }
  }

  // rect11 (animate)
  if (insideBox(vUv, vec2(rect11.x / dims.x, rect11.y / dims.y), vec2((rect11.x + rect11.z) / dims.x, (rect11.y + rect11.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect11.x;
    float y = floor(vUv.y * dims.y) - rect11.y;

    vec2 pos = vec2(vUv.x, vUv.y + sin(x * 0.1 + time * 2.0) / dims.y * 1.0);
    rgb = texture2D(buffer, pos).rgb;
  }

  // rect12 (reset)
  if (insideBox(vUv, vec2(rect12.x / dims.x, rect12.y / dims.y), vec2((rect12.x + rect12.z) / dims.x, (rect12.y + rect12.w) / dims.y)) >= 1.0) {
    if (random(vec2(vUv.x + time, vUv.y)) < 0.06) {
      rgb = texture2D(tex, vUv).rgb;
    }
  }

  // rect13 (reset)
  if (insideBox(vUv, vec2(rect13.x / dims.x, rect13.y / dims.y), vec2((rect13.x + rect13.z) / dims.x, (rect13.y + rect13.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect13.x;
    float y = floor(vUv.y * dims.y) - rect13.y;

    if (mod(x + y, 4.0) == mod(floor(time * 10.0), 4.0) && mod(y, 2.0) == 0.0) {
      rgb = texture2D(tex, vUv).rgb;
    }

    if (mod(x + y + 1.0, 4.0) == mod(floor(time * 6.0), 4.0) && mod(y, 2.0) == 1.0) {
      rgb = texture2D(tex, vUv).rgb;
    }
  }

  // rect14 (reset)
  if (insideBox(vUv, vec2(rect14.x / dims.x, rect14.y / dims.y), vec2((rect14.x + rect14.z) / dims.x, (rect14.y + rect14.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect14.x;
    float y = floor(vUv.y * dims.y) - rect14.y;
    float step = mod(floor(time * 50.0), 16.0);

    if (mod(x, 16.0) == step) {
      rgb = texture2D(tex, vUv).rgb;
    }
  }

  // rect15 (reset)
  if (insideBox(vUv, vec2(rect15.x / dims.x, rect15.y / dims.y), vec2((rect15.x + rect15.z) / dims.x, (rect15.y + rect15.w) / dims.y)) >= 1.0) {
    float x = floor(vUv.x * dims.x) - rect15.x;
    float y = floor(vUv.y * dims.y) - rect15.y;
    float step = mod(floor(time * 50.0), 16.0);

    if (mod(y, 16.0) == step) {
      rgb = texture2D(tex, vUv).rgb;
    }
  }

  vec4 color = vec4(rgb, 1.0);

  gl_FragColor = color;
}
