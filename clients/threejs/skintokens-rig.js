// SkinTokens rig payloads for three.js: attach the rig the server sends to the
// scene GLTFLoader loaded from the file that was uploaded.
//
//   import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
//   import { decodeRigPayload, applyRigPayload } from './skintokens-rig.js';
//
//   const gltf = await new GLTFLoader().parseAsync(uploadedArrayBuffer, '');
//   const rig = await decodeRigPayload(payloadArrayBuffer);  // binary WebSocket frame
//   const { skeleton, armature } = await applyRigPayload(gltf, rig);
//   scene.add(gltf.scene);
//
// The meshes keep their materials, UVs and everything else GLTFLoader loaded;
// each triangle mesh becomes a THREE.SkinnedMesh bound to the new skeleton. The
// model keeps the file's own placement, like the Python client's apply_rig
// (pass { ground: true } to stand it on the ground plane like the server's GLB).
//
// Payload format: src/rig_package/glb_io/rig_payload.py (and the README).
// Requires three r150+ and DecompressionStream (all current browsers, Node 18+).

import { Bone, BufferAttribute, Group, Matrix4, Skeleton, SkinnedMesh } from 'three';

const MAGIC = 0x47524b53; // "SKRG", little-endian
const VERSION = 1;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const INDEX_TYPES = { uint8: Uint8Array, uint16: Uint16Array };
const WEIGHT_TYPES = { uint16: Uint16Array, float32: Float32Array };

function fail(message) {
  throw new Error(`invalid rig payload: ${message}`);
}

async function inflate(bytes) {
  // "deflate" is the zlib format (RFC 1950), which the server writes.
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function typedArray(raw, spec, types, name, count) {
  const Type = types[spec && spec.component_type];
  if (!Type) fail(`${name}.component_type must be one of ${Object.keys(types)}`);
  if (spec.byte_length !== count * Type.BYTES_PER_ELEMENT) fail(`${name}.byte_length does not match`);
  if (spec.byte_offset % Type.BYTES_PER_ELEMENT || spec.byte_offset + spec.byte_length > raw.byteLength) {
    fail(`${name} is out of bounds`);
  }
  return new Type(raw.buffer, raw.byteOffset + spec.byte_offset, count);
}

/**
 * Parse a rig payload (ArrayBuffer or typed array).
 * @returns {Promise<{header: object, jointIndices: Uint8Array|Uint16Array,
 *   jointWeights: Uint16Array|Float32Array}>} (N x K row-major arrays, K = header.influences)
 */
export async function decodeRigPayload(data) {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 28 || view.getUint32(0, true) !== MAGIC) fail('not a rig payload (bad magic)');
  if (view.getUint32(4, true) !== VERSION) fail(`unsupported version ${view.getUint32(4, true)}`);
  if (view.getUint32(8, true) !== bytes.byteLength) fail('declared length does not match');

  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== CHUNK_JSON || 20 + jsonLength + 8 > bytes.byteLength) fail('bad JSON chunk');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
  if (header.format !== 'skintokens-rig') fail('format is not "skintokens-rig"');

  const binStart = 20 + jsonLength;
  const binLength = view.getUint32(binStart, true);
  if (view.getUint32(binStart + 4, true) !== CHUNK_BIN || binStart + 8 + binLength > bytes.byteLength) {
    fail('bad BIN chunk');
  }
  const { compression, byte_length: rawLength, stored_byte_length: storedLength } = header.bin || {};
  if (!(storedLength <= binLength)) fail('BIN chunk shorter than bin.stored_byte_length');
  const stored = bytes.subarray(binStart + 8, binStart + 8 + storedLength);
  let raw;
  if (compression === 'zlib') raw = await inflate(stored);
  else if (compression === 'none') raw = stored.slice(); // aligned copy
  else fail(`unsupported compression ${compression}`);
  if (raw.byteLength !== rawLength) fail('BIN data does not match bin.byte_length');

  const J = header.joint_names.length;
  if (header.parents.length !== J || header.joint_heads.length !== J) fail('joint arrays differ in length');
  const count = header.vertex_count * header.influences;
  const jointIndices = typedArray(raw, header.joint_indices, INDEX_TYPES, 'joint_indices', count);
  const jointWeights = typedArray(raw, header.joint_weights, WEIGHT_TYPES, 'joint_weights', count);
  for (let i = 0; i < count; i++) if (jointIndices[i] >= J) fail('joint index out of range');
  return { header, jointIndices, jointWeights };
}

// The three.js objects of glTF node `nodeIndex`'s triangle primitives, by
// primitive index. GLTFLoader's associations name each primitive's object
// reliably, but not always its node (instances of one mesh share a mapping),
// so the node's object comes from the parser's cache (the very object in
// gltf.scene) and only its own subtree, not its child nodes, is searched.
async function primitiveObjects(parser, nodeIndex, nodeObjects) {
  const nodeDef = parser.json.nodes[nodeIndex];
  const root = await parser.getDependency('node', nodeIndex);
  const found = new Map();
  const visit = (o) => {
    const a = parser.associations.get(o);
    if (o.isMesh && a && a.meshes === nodeDef.mesh && a.primitives !== undefined) found.set(a.primitives, o);
    for (const child of o.children) if (!nodeObjects.has(child)) visit(child);
  };
  visit(root);
  return found;
}

/**
 * Skin the meshes of `gltf` (GLTFLoader's result for the uploaded file) with a
 * decoded rig payload. Adds an armature Group (the bones) to `gltf.scene`.
 * @param {object} gltf
 * @param {{header, jointIndices, jointWeights}} rig from decodeRigPayload
 * @param {{ground?: boolean}} [options] ground (default false): show the model on
 *   the ground plane like the server's GLB, i.e. shifted by header.ground_offset
 * @returns {Promise<{skeleton: Skeleton, armature: Group, meshes: SkinnedMesh[]}>}
 */
export async function applyRigPayload(gltf, rig, { ground = false } = {}) {
  const { header, jointIndices, jointWeights } = rig;
  const K = header.influences;
  if (K !== 4) throw new Error(`three.js skins 4 influences per vertex; the payload has ${K}`);
  const parser = gltf.parser;

  // Every (node, primitive) the payload has rows for, found before anything changes.
  const nodeObjects = new Set(await Promise.all(
    (parser.json.nodes || []).map((_, i) => parser.getDependency('node', i))));
  const targets = [];
  for (const entry of header.layout) {
    const found = await primitiveObjects(parser, entry.node, nodeObjects);
    for (const prim of entry.primitives) {
      const mesh = found.get(prim.index);
      if (!mesh) throw new Error(`rig payload: no mesh for node ${entry.node} primitive ${prim.index}`);
      targets.push({ entry, prim, mesh });
    }
  }

  const offset = ground ? header.ground_offset : [0, 0, 0];
  gltf.scene.updateMatrixWorld(true);
  // The server's space is gltf.scene's own: shift there, even if the app has
  // already placed gltf.scene somewhere.
  const sceneWorld = gltf.scene.matrixWorld.clone();
  const shift = sceneWorld.clone()
    .multiply(new Matrix4().makeTranslation(offset[0], offset[1], offset[2]))
    .multiply(sceneWorld.clone().invert());

  // Bones: joint heads are in the glTF scene's world space, as is gltf.scene's content.
  const armature = new Group();
  armature.name = header.armature_name;
  const heads = header.joint_heads.map((h) => h.map((x, i) => x + offset[i]));
  const bones = header.joint_names.map((name) => Object.assign(new Bone(), { name }));
  header.parents.forEach((p, j) => {
    const h = heads[j];
    const base = p >= 0 ? heads[p] : [0, 0, 0];
    bones[j].position.set(h[0] - base[0], h[1] - base[1], h[2] - base[2]);
    (p >= 0 ? bones[p] : armature).add(bones[j]);
  });
  gltf.scene.add(armature);
  gltf.scene.updateMatrixWorld(true);
  const skeleton = new Skeleton(bones); // inverses from the bind pose just set

  const seenGeometry = new Set();
  const meshes = [];
  let row = 0;
  for (const { entry, prim, mesh } of targets) {
    let geometry = mesh.geometry;
    if (geometry.attributes.position.count !== prim.vertex_count) {
      throw new Error(`rig payload does not match this file (node ${entry.node} primitive ${prim.index}: ` +
        `${geometry.attributes.position.count} vertices, payload ${prim.vertex_count})`);
    }
    // A mesh used by several nodes shares its geometry; each use has its own skin.
    if (seenGeometry.has(geometry)) geometry = geometry.clone();
    seenGeometry.add(geometry);
    const start = row * K;
    const end = (row + prim.vertex_count) * K;
    geometry.setAttribute('skinIndex', new BufferAttribute(jointIndices.slice(start, end), 4));
    geometry.setAttribute('skinWeight', new BufferAttribute(jointWeights.slice(start, end), 4,
      header.joint_weights.normalized));
    row += prim.vertex_count;

    const skinned = new SkinnedMesh(geometry, mesh.material);
    skinned.name = mesh.name;
    skinned.position.copy(mesh.position);
    skinned.quaternion.copy(mesh.quaternion);
    skinned.scale.copy(mesh.scale);
    skinned.userData = mesh.userData;
    const parent = mesh.parent;
    parent.add(skinned);
    parent.remove(mesh);
    for (const child of [...mesh.children]) skinned.add(child);
    skinned.updateMatrixWorld(true);
    // The server skinned the vertices as placed in the scene (a node that was
    // skinned in the input: as stored, in the scene's space), then shifted
    // them with the joints.
    const placed = entry.skinned_input ? sceneWorld : skinned.matrixWorld;
    const bindMatrix = shift.clone().multiply(placed);
    skinned.bind(skeleton, bindMatrix);
    meshes.push(skinned);
  }
  if (row !== header.vertex_count) throw new Error('rig payload: layout does not cover vertex_count');
  return { skeleton, armature, meshes };
}
