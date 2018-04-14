// Aldaron's Device Interface / GPU / Vulkan
// Copyright (c) 2017-2018 Jeron Lau <jeron.lau@plopgrizzly.com>
// Licensed under the MIT LICENSE
//
// src/lib.rs

//! Vulkan implementation for adi_gpu.

// #![no_std]

#[macro_use]
extern crate ami;
extern crate awi;
extern crate afi;
extern crate asi_vulkan;
extern crate adi_gpu_base;
extern crate libc;

/// Transform represents a transformation matrix.
pub(crate) mod renderer;

pub use base::Shape;
pub use base::Gradient;
pub use base::Model;
pub use base::TexCoords;
pub use renderer::Texture;

use ami::*;
use adi_gpu_base as base;
use adi_gpu_base::ShapeHandle;

/// To render anything with adi_gpu, you have to make a `Display`
pub struct Display {
	window: awi::Window,
	renderer: renderer::Renderer,
}

impl base::Display for Display {
	type Texture = Texture;

	fn new(title: &str, icon: &afi::Graphic) -> Option<Self> {
		let window = awi::Window::new(title, &icon, None);
		let renderer = renderer::Renderer::new("ADI Application",
			window.get_connection(), (0.0, 0.0, 0.0));

		Some(Display { window, renderer })
	}

	fn color(&mut self, color: (f32, f32, f32)) {
		self.renderer.bg_color(color);
	}

	fn update(&mut self) -> Option<awi::Input> {
		if let Some(input) = self.window.update() {
			return Some(input);
		}

		// Update Window:
		self.renderer.update();
		// Return None, there was no input, updated screen.
		None
	}

	fn camera(&mut self, xyz: (f32,f32,f32), rotate_xyz: (f32,f32,f32)) {
		self.renderer.set_camera(xyz, rotate_xyz);
		self.renderer.camera();
	}

	fn model(&mut self, vertices: &[f32], indices: &[u32]) -> Model {
		Model(self.renderer.model(vertices, indices))
	}

	fn fog(&mut self, fog: Option<(f32, f32)>) -> () {
		if let Some(fog) = fog {
			self.renderer.fog(fog);
		} else {
			self.renderer.fog((::std::f32::MAX, 0.0));
		}
	}

	fn texture(&mut self, graphic: afi::Graphic) -> Texture {
		let (w, h, pixels) = graphic.as_slice();

		self.renderer.texture(w, h, pixels)
	}

	fn gradient(&mut self, colors: &[f32]) -> Gradient {
		Gradient(self.renderer.colors(colors))
	}

	fn texcoords(&mut self, texcoords: &[f32]) -> TexCoords {
		TexCoords(self.renderer.texcoords(texcoords))
	}

	fn set_texture(&mut self, texture: &mut Self::Texture, pixels: &[u32]) {
		self.renderer.set_texture(texture, pixels);
	}

	#[inline(always)]
	fn shape_solid(&mut self, model: &Model, transform: Mat4,
		color: [f32; 4], blending: bool, fancy: bool, fog: bool,
		camera: bool) -> Shape
	{
		base::new_shape(self.renderer.solid(model.0, transform.0, color,
			blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_gradient(&mut self, model: &Model, transform: Mat4,
		colors: Gradient, blending: bool, fancy: bool, fog: bool,
		camera: bool) -> Shape
	{
		base::new_shape(self.renderer.gradient(model.0, transform.0,
			colors.0, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_texture(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, blending: bool, fancy: bool,
		fog: bool, camera: bool) -> Shape
	{
		base::new_shape(self.renderer.textured(model.0, transform.0,
			texture, tc.0, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_faded(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, alpha: f32, fancy: bool,
		fog: bool, camera: bool) -> Shape
	{
		base::new_shape(self.renderer.faded(model.0, transform.0,
			texture, tc.0, alpha, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_tinted(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, tint: [f32; 4], blending: bool,
		fancy: bool, fog: bool, camera: bool) -> Shape
	{
		base::new_shape(self.renderer.tinted(model.0, transform.0,
			texture, tc.0, tint, blending, fancy, fog, camera))
	}

	#[inline(always)]
	fn shape_complex(&mut self, model: &Model, transform: Mat4,
		texture: Texture, tc: TexCoords, tints: Gradient,
		blending: bool, fancy: bool, fog: bool, camera: bool) -> Shape
	{
		base::new_shape(self.renderer.complex(model.0, transform.0,
			texture, tc.0, tints.0, blending, fancy, fog, camera))
	}

	fn transform(&mut self, shape: &mut Shape, transform: &Mat4) {
		self.renderer.transform(&mut base::get_shape(shape), transform);
	}

	fn resize(&mut self, wh: (u32, u32)) -> () {
		self.renderer.resize(wh);
	}

	fn wh(&self) -> (u32, u32) {
		self.window.wh()
	}
}

impl base::Texture for Texture {
	/// Get the width and height.
	fn wh(&self) -> (u32, u32) {
		(self.w, self.h)
	}
}
