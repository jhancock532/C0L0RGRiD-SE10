//P
precision mediump float;
#define GLSLIFY 1

uniform sampler2D buffer;
varying vec2 vUv;
void main() { 
  gl_FragColor = texture2D(buffer, vUv);
}

//S
#define GLSLIFY 1
attribute vec2 position;

varying vec2 vUv;

void main() {
  gl_Position = vec4(position, 0.0, 1.0);
  vUv = 0.5 * (position + 1.0);
}
