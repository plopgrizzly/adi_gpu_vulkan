// "adi_gpu_vulkan" crate - Licensed under the MIT LICENSE
//  * Copyright (c) 2018  Jeron A. Lau <jeron.lau@plopgrizzly.com>

use std::{ mem };

// use awi::Window;
use adi_gpu_base as base;
use adi_gpu_base::WindowConnection;

mod ffi;

use asi_vulkan;
use asi_vulkan::types::*;
use asi_vulkan::Image;
use asi_vulkan::Style;
use asi_vulkan::Buffer;

// TODO
use asi_vulkan::TransformUniform;
use asi_vulkan::FogUniform;
use asi_vulkan::Sprite;
use asi_vulkan::Vk;

use ami::{ Mat4, IDENTITY };

use ShapeHandle;

#[derive(Clone)] #[repr(C)] struct TransformFullUniform {
	mat4: [f32; 16],
	hcam: u32,
}

#[derive(Clone)] #[repr(C)] struct TransformAndFadeUniform {
	mat4: [f32; 16],
	fade: f32,
	hcam: u32,
}

#[derive(Clone)] #[repr(C)] struct TransformAndColorUniform {
	mat4: [f32; 16],
	vec4: [f32; 4],
	hcam: u32,
}

pub struct Vw {
	connection: Vk,
	present_queue: VkQueue,
	swapchain: VkSwapchainKHR,
	width:u32, height:u32, // Swapchain Dimensions.
	present_images: [VkImage; 2], // 2 for double-buffering
	frame_buffers: [VkFramebuffer; 2], // 2 for double-buffering
	color_format: VkFormat,
	image_count: u32, // 1 (single-buffering) or 2 (double-buffering)
	submit_fence: asi_vulkan::Fence, // The submit fence
	present_image_views: [VkImageView; 2], // 2 for double-buffering
	ms_image: Image,
	depth_image: Image,
	render_pass: VkRenderPass,
	next_image_index: u32,
	present_mode: VkPresentModeKHR,
}

/// A texture on the GPU.
pub struct Texture {
	mappable_image: Image,
	image: Option<Image>,
//	view: VkImageView,
	pub(super) w: u32,
	pub(super) h: u32,
	pitch: u32,
	staged: bool,
}

pub struct Shape {
	num_buffers: usize,
	buffers: [VkBuffer; 3],
	instance: Sprite,
	bbox: ::ami::BBox,
	model: usize,
	fans: Vec<(u32, u32)>,
}

pub struct Model {
	shape: asi_vulkan::Buffer,
	vertex_count: u32,
	points: Vec<f32>,
	fans: Vec<(u32, u32)>,
}

pub struct TexCoords {
	vertex_buffer: Buffer,
	vertex_count: u32,
}

pub struct Gradient {
	vertex_buffer: Buffer,
	vertex_count: u32,
}

impl ::ami::Collider for Shape {
	fn bbox(&self) -> ::ami::BBox {
		self.bbox
	}
}

impl Shape {
// TODO
/*	pub fn animate(window: &mut Window, index: usize, i: usize,
		texture: *const NativeTexture, style: Style)
	{
		let hastx = window.sprites[index].hastx;

		// Must be same style
		if hastx {
			if (texture as *const _ as usize) == 0 {
				panic!("Can't set Style of a Sprite initialized\
					with Style::Texture to Style::Solid");
			}
		} else {
			if (texture as *const _ as usize) != 0 {
				panic!("Can't set Style of a Sprite initialized\
					with Style::Solid to Style::Texture");
			}
		}

		// Free old Style, and set new uniform buffers.
		unsafe {
			asi_vulkan::destroy_uniforms(&window.vw, &mut
				window.sprites[index].instances[i].instance);
			window.sprites[index].instances[i].instance =
				vw_vulkan_uniforms(&window.vw, style, texture,
					if hastx { 1 } else { 0 });
		}
		// TODO: Optimize when using same value from vw_vulkan_uniforms
		// Set texture
//		unsafe {
//			vw_vulkan_txuniform(&window.vw,
//				&mut window.sprites[index].shape.instances[i].instance, texture,
//				if window.sprites[index].shape.hastx { 1 } else { 0 });
//		}
		Shape::enable(window, index, i, true);
	}

	pub fn vertices(window: &Window, index: usize, v: &[f32]) {
		ffi::copy_memory(window.vw.device,
			window.sprites[index].shape.vertex_buffer_memory, v);
	}*/
}

fn swapchain_resize(vw: &mut Vw) {
	unsafe {
		// Link swapchain to vulkan instance.
		asi_vulkan::create_swapchain(
			&mut vw.connection,
			&mut vw.swapchain,
			vw.width,
			vw.height,
			&mut vw.image_count,
			vw.color_format.clone(),
			vw.present_mode.clone(),
			&mut vw.present_images[0]);

		// Link Image Views for each framebuffer
		vw.submit_fence = asi_vulkan::create_image_view(
			&mut vw.connection,
			&vw.color_format,
			vw.image_count,
			&mut vw.present_images,
			&mut vw.present_image_views,
			vw.present_queue,
		);

		// Link Depth Buffer to swapchain
		let img = asi_vulkan::create_depth_buffer(
			&mut vw.connection,
			&vw.submit_fence,
			vw.present_queue,
			vw.width,
			vw.height,
		);

		vw.depth_image = img;

		// Create multisampling buffer
		let img = asi_vulkan::create_ms_buffer(
			&mut vw.connection,
			&vw.color_format,
			vw.width,
			vw.height,
		);

		vw.ms_image = img;

		// Link Render Pass to swapchain
		vw.render_pass = asi_vulkan::create_render_pass(
			&mut vw.connection,
			&vw.color_format,
		);

		// Link Framebuffers to swapchain
		asi_vulkan::create_framebuffers(
			&mut vw.connection,
			vw.image_count,
			vw.render_pass,
			&vw.present_image_views,
			&vw.ms_image,
			&vw.depth_image,
			vw.width,
			vw.height,
			&mut vw.frame_buffers,
		);
	}
}

fn swapchain_delete(vw: &mut Vw) {
	unsafe {
		asi_vulkan::destroy_swapchain(
			&mut vw.connection,
			&vw.frame_buffers,
			&vw.present_image_views,
			vw.render_pass,
			vw.image_count,
			vw.swapchain,
		);
	}
}

fn new_texture(vw: &mut Vw, width: u32, height: u32) -> Texture {
//	let mut format_props = unsafe { mem::uninitialized() };
	let staged = !vw.connection.sampled();

	let mappable_image = asi_vulkan::Image::new(
		&mut vw.connection, width, height, VkFormat::R8g8b8a8Srgb,
		VkImageTiling::Linear,
		if staged { VkImageUsage::TransferSrcBit }
		else { VkImageUsage::SampledBit },
		VkImageLayout::Preinitialized,
		0x00000006 /* visible|coherent */,
		VkSampleCount::Sc1
	);

	let layout = unsafe {
		asi_vulkan::subres_layout(&mut vw.connection, &mappable_image)
	};

	let pitch = layout.row_pitch;

	let image = if staged {
		Some(asi_vulkan::Image::new(
			&mut vw.connection, width, height,
			VkFormat::R8g8b8a8Srgb,
			VkImageTiling::Optimal,
			VkImageUsage::TransferDstAndUsage,
			VkImageLayout::Undefined, 0,
			VkSampleCount::Sc1))
	} else {
		None
	};

/*	let view = unsafe {
		asi_vulkan::create_imgview(&mut vw.connection,
			image.as_ref().unwrap_or(&mappable_image),
			VkFormat::R8g8b8a8Srgb,
			true
		)
	};*/

	Texture {
		staged, mappable_image,	image, pitch: pitch as u32,
		w: width, h: height,
	}
}

fn set_texture(vw: &mut Vw, texture: &mut Texture, rgba: &[u32]) {
//	if texture.pitch != 4 {
		ffi::copy_memory_pitched(&mut vw.connection,
			texture.image
				.as_ref()
				.unwrap_or(&texture.mappable_image)
				.memory(),
			rgba, texture.w as isize,
			texture.h as isize, texture.pitch as isize);
//	} else {
//		ffi::copy_memory(connection, vw.device, texture.memory,
//			rgba.as_ptr(), mem::size_of::<u32>() * rgba.len());
/*	}*/

	if texture.staged {
		// Use optimal tiled image - create from linear tiled image

		// Copy data from linear image to optimal image.
		unsafe {
			asi_vulkan::copy_image(&mut vw.connection,
				&texture.mappable_image,
				texture.image.as_ref().unwrap(),
				texture.w, texture.h
			);
		}
	} else {
		// Use a linear tiled image for the texture, is supported
		texture.image = None;
	}
}

/*pub fn make_styles(vw: &mut Vw, extrashaders: &[Shader], shaders: &mut Vec<Style>)
{
	let mut shadev = Vec::new();
	let default_shaders = [
//		Shader::create(vw, include_bytes!("res/texture-vert.spv"),
//			include_bytes!("res/texture-frag.spv"), 1),
	];
	shadev.extend(default_shaders.iter().cloned());
	shadev.extend(extrashaders.iter().cloned());

	*shaders = vec![Style { pipeline: 0, descsetlayout: 0,
		pipeline_layout: 0 }; shadev.len()];
	unsafe {
		vw_vulkan_pipeline(&mut shaders[0], vw, &shadev[0],
			shadev.len() as u32);
	}
}*/

impl Vw {
	pub fn new(window_connection: WindowConnection)
		-> Result<Vw, &'static str>
	{
		// START BLOCK: TODO: this should all be condensed to one
		// asi_vulkan function for safety.
		let mut connection = ffi::vulkan::Vulkan::new()? .0;

		ffi::create_surface::create_surface(&mut connection,
			window_connection);
		unsafe {
			asi_vulkan::get_gpu(&mut connection)?;
			asi_vulkan::create_device(&mut connection);
		}
		// END BLOCK
		let present_queue = unsafe {
			asi_vulkan::create_queue(&mut connection)
		};
		unsafe {
			asi_vulkan::create_command_buffer(&mut connection);
		}
		// END BLOCK 2
		let color_format = unsafe {
			asi_vulkan::get_color_format(&mut connection)
		};
		let mut image_count = unsafe {
			asi_vulkan::get_buffering(&mut connection)
		};
		let present_mode = unsafe {
			asi_vulkan::get_present_mode(&mut connection)
		};
		// Prepare Swapchain (TODO: is duplicate of swapchain_resize).
		let mut swapchain = unsafe { mem::zeroed() };
		let mut present_images: [VkImage; 2] = unsafe { mem::zeroed() };
		let mut present_image_views = [unsafe { mem::zeroed() }; 2];
		let mut frame_buffers: [VkFramebuffer; 2]
			= unsafe { mem::uninitialized() };
		let width = 640; // TODO w
		let height = 360; // TODO h

		unsafe {
			// Link swapchain to vulkan instance.
			asi_vulkan::create_swapchain(
				&mut connection,
				&mut swapchain,
				width,
				height,
				&mut image_count,
				color_format.clone(),
				present_mode.clone(),
				&mut present_images[0]);
		}

		let submit_fence = unsafe {
			// Link Image Views for each framebuffer
			asi_vulkan::create_image_view(
				&mut connection,
				&color_format,
				image_count,
				&mut present_images,
				&mut present_image_views,
				present_queue,
			)
		};
		// Link Depth Buffer to swapchain
		let depth_image = unsafe {
			asi_vulkan::create_depth_buffer(
				&mut connection,
				&submit_fence,
				present_queue,
				width, height,
			)
		};

		// Create multisampling buffer
		let ms_image = unsafe {
			asi_vulkan::create_ms_buffer(
				&mut connection,
				&color_format,
				width, height,
			)
		};

		// Link Render Pass to swapchain
		let render_pass = unsafe {
			asi_vulkan::create_render_pass(
				&mut connection,
				&color_format,
			)
		};

		// Link Framebuffers to swapchain
		unsafe {
			asi_vulkan::create_framebuffers(
				&mut connection,
				image_count,
				render_pass,
				&present_image_views,
				&ms_image,
				&depth_image,
				width, height,
				&mut frame_buffers,
			);
		}

		// Finish connection with the texture sampler (TODO: in block2).
		unsafe {
			asi_vulkan::new_sampler(&mut connection);
		}

		Ok(Vw {
			connection, present_queue, swapchain,
			width, height, present_images, frame_buffers,
			color_format, image_count, submit_fence,
			present_image_views, ms_image, depth_image, render_pass,
			next_image_index: 0, present_mode,
		})
	}
}

fn draw_shape(connection: &mut Vk, shape: &Shape) {
	unsafe {
		for i in shape.fans.iter() {
			asi_vulkan::cmd_bind_vb(connection,
				&shape.buffers[..shape.num_buffers]);
			asi_vulkan::cmd_bind_pipeline(connection,
				shape.instance.pipeline);
			asi_vulkan::cmd_bind_descsets(connection,
				shape.instance.pipeline_layout,
				shape.instance.handles().0/*desc_set*/);

			asi_vulkan::cmd_draw(connection, i.1,
				1, i.0, 0);
		}
	}
}

pub struct Renderer {
	vw: Vw,
	ar: f64,
	opaque_octree: ::ami::Octree<Shape>,
	alpha_octree: ::ami::Octree<Shape>,
	gui_vec: Vec<Shape>,
	opaque_sorted: Vec<u32>,
	alpha_sorted: Vec<u32>,
//	opaque_points: ::ami::Points,
//	alpha_points: ::ami::Points,
//	opaque_shapes: Vec<Shape>,
//	alpha_shapes: Vec<Shape>,
	models: Vec<Model>,
	texcoords: Vec<TexCoords>,
	gradients: Vec<Gradient>,
	textures: Vec<Texture>,
	style_solid: Style,
	style_nasolid: Style,
	style_texture: Style,
	style_natexture: Style,
	style_gradient: Style,
	style_nagradient: Style,
	style_faded: Style,
	style_tinted: Style,
	style_natinted: Style,
	style_complex: Style,
	style_nacomplex: Style,
	projection: ::Mat4,
	camera_memory: asi_vulkan::Memory<TransformUniform>,
	effect_memory: asi_vulkan::Memory<FogUniform>,
	clear_color: (f32, f32, f32),
	frustum: ::ami::Frustum,
	xyz: (f32,f32,f32),
	rotate_xyz: (f32,f32,f32),
}

impl Renderer {
	pub fn new(window_connection: WindowConnection,
		clear_color: (f32, f32, f32)) -> Result<Renderer, &'static str>
	{
		let mut vw = Vw::new(window_connection)?;

		let solid_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/solid-vert.spv"));
		let solid_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/solid-frag.spv"));
		let texture_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/texture-vert.spv"));
		let texture_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/texture-frag.spv"));
		let gradient_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-vert.spv"));
		let gradient_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-frag.spv"));
		let faded_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/faded-vert.spv"));
		let faded_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/faded-frag.spv"));
		let tinted_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-vert.spv"));
		let tinted_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-frag.spv"));
		let complex_vert = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-vert.spv"));
		let complex_frag = asi_vulkan::ShaderModule::new(
			&mut vw.connection, include_bytes!(
			"../shaders/res/gradient-frag.spv"));
		let style_solid = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&solid_vert, &solid_frag, 0, 1, true);
		let style_nasolid = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&solid_vert, &solid_frag, 0, 1, false);
		let style_texture = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&texture_vert, &texture_frag, 1, 2, true);
		let style_natexture = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&texture_vert, &texture_frag, 1, 2, false);
		let style_gradient = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&gradient_vert, &gradient_frag, 0, 2, true);
		let style_nagradient = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&gradient_vert, &gradient_frag, 0, 2, false);
		let style_faded = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&faded_vert, &faded_frag, 1, 2, true);
		let style_tinted = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&tinted_vert, &tinted_frag, 1, 2, true);
		let style_natinted = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&tinted_vert, &tinted_frag, 1, 2, false);
		let style_complex = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&complex_vert, &complex_frag, 1, 3, true);
		let style_nacomplex = Style::new(&mut vw.connection,
			vw.render_pass, vw.width, vw.height,
			&complex_vert, &complex_frag, 1, 3, false);

		let ar = vw.width as f64 / vw.height as f64;
		let projection = ::base::projection(ar, 90.0);
		let (camera_memory, effect_memory) = unsafe {
			asi_vulkan::vw_camera_new(&mut vw.connection,
				(clear_color.0, clear_color.1, clear_color.2,
					1.0), (::std::f32::MAX, ::std::f32::MAX))
		};

		let mut renderer = Renderer {
			vw, ar, projection,
			camera_memory, effect_memory,
			alpha_octree: ::ami::Octree::new(),
			opaque_octree: ::ami::Octree::new(),
			gui_vec: Vec::new(),
			opaque_sorted: Vec::new(),
			alpha_sorted: Vec::new(),
			gradients: Vec::new(),
			models: Vec::new(),
			texcoords: Vec::new(),
			textures: Vec::new(),
			style_solid, style_nasolid,
			style_texture, style_natexture,
			style_gradient, style_nagradient,
			style_faded,
			style_tinted, style_natinted,
			style_complex, style_nacomplex,
			clear_color,
			frustum: ::ami::Frustum::new(
				::ami::Vec3::new(0.0, 0.0, 0.0),
				100.0 /* TODO: Based on fog.0 + fog.1 */, 90.0,
				(2.0 * ((45.0 * ::std::f64::consts::PI / 180.0).tan() / ar).atan()) as f32,
				0.0, 0.0), // TODO: FAR CLIP PLANE
			xyz: (0.0, 0.0, 0.0),
			rotate_xyz: (0.0, 0.0, 0.0),
		};

		renderer.camera();

		Ok(renderer)
	}

	pub fn bg_color(&mut self, rgb: (f32, f32, f32)) {
		self.clear_color = rgb;
	}

	pub fn update(&mut self) {
		let matrix = IDENTITY
			.rotate(self.rotate_xyz.0, self.rotate_xyz.1,
				self.rotate_xyz.2)
			.translate(self.xyz.0, self.xyz.1, self.xyz.2);

		let mut presenting_complete_sem = unsafe {
			asi_vulkan::new_semaphore(&mut self.vw.connection)
		};

		let rendering_complete_sem = unsafe {
			asi_vulkan::new_semaphore(&mut self.vw.connection)
		};

		unsafe {
			self.vw.next_image_index = asi_vulkan::get_next_image(
				&mut self.vw.connection,
				&mut presenting_complete_sem,
				self.vw.swapchain,
			);

			asi_vulkan::draw_begin(&mut self.vw.connection,
				self.vw.render_pass,
				self.vw.present_images[self.vw.next_image_index as usize],
				self.vw.frame_buffers[self.vw.next_image_index as usize],
				self.vw.width,
				self.vw.height,
				self.clear_color.0, self.clear_color.1,
				self.clear_color.2
			);
		}

		let frustum = matrix * self.frustum;

//		self.opaque_octree.print();
//		println!("FRUSTUM {:?}", frustum);

		self.opaque_octree.nearest(&mut self.opaque_sorted, frustum);
		for id in self.opaque_sorted.iter() {
//			println!("drawing opaque....");

			let shape = &self.opaque_octree[*id];

			draw_shape(&mut self.vw.connection, shape);
		}

		self.alpha_octree.farthest(&mut self.alpha_sorted, frustum);
		for id in self.alpha_sorted.iter() {
//			println!("drawing alpha....");
			let shape = &self.alpha_octree[*id];

			draw_shape(&mut self.vw.connection, shape);
		}

		for shape in self.gui_vec.iter() {
//			println!("drawing gui....");
			draw_shape(&mut self.vw.connection, shape);
		}

		unsafe {
			asi_vulkan::end_render_pass(&mut self.vw.connection);

			asi_vulkan::pipeline_barrier(&mut self.vw.connection,
				self.vw.present_images[self.vw.next_image_index as usize]);

			asi_vulkan::end_cmdbuff(&mut self.vw.connection);

			{ // Drop when it's done use
			let fence = asi_vulkan::Fence::new(&mut self.vw.connection);

			asi_vulkan::queue_submit(&mut self.vw.connection,
				&fence,
				VkPipelineStage::BottomOfPipe,
				self.vw.present_queue,
				Some(rendering_complete_sem));
				
			asi_vulkan::wait_fence(&mut self.vw.connection, &fence);
			}
				
			asi_vulkan::queue_present(&mut self.vw.connection,
				self.vw.present_queue,
				rendering_complete_sem,
				self.vw.swapchain,
				self.vw.next_image_index);

			asi_vulkan::drop_semaphore(&mut self.vw.connection,
				rendering_complete_sem);

			asi_vulkan::drop_semaphore(&mut self.vw.connection,
				presenting_complete_sem);

			asi_vulkan::wait_idle(&mut self.vw.connection);
		}
	}

	pub fn resize(&mut self, size: (u32, u32)) {
		self.vw.width = size.0;
		self.vw.height = size.1;
		self.ar = size.0 as f64 / size.1 as f64;
		self.frustum = ::ami::Frustum::new(
			self.frustum.center,
			self.frustum.radius,
			90.0, (2.0 * ((45.0 * ::std::f64::consts::PI / 180.0)
				.tan() / self.ar).atan()) as f32,
			self.frustum.xrot, self.frustum.yrot);

		swapchain_delete(&mut self.vw);
		swapchain_resize(&mut self.vw);

		self.projection = ::base::projection(self.ar, 90.0);
		self.camera();
	}

	pub fn texture(&mut self, width: u32, height: u32, rgba: &[u32])
		-> usize
	{
		let mut texture = new_texture(&mut self.vw, width, height);

		set_texture(&mut self.vw, &mut texture, rgba);

		let a = self.textures.len();
		self.textures.push(texture);
		a
	}

	pub fn set_texture(&mut self, texture: usize, rgba: &[u32]) {
		set_texture(&mut self.vw, &mut self.textures[texture], rgba);
	}

	/// Push a model (collection of vertices) into graphics memory.
	pub fn model(&mut self, vertices: &[f32], fans: Vec<(u32, u32)>)
		-> usize
	{
		let shape = unsafe {
			asi_vulkan::new_buffer(&mut self.vw.connection,
				vertices)
		};

		let a = self.models.len();

		let points = vertices.to_vec();

		self.models.push(Model {
			shape,
			vertex_count: vertices.len() as u32 / 4,
			points, fans,
		});

		a
	}

	/// Push texture coordinates (collection of vertices) into graphics
	/// memory.
	pub fn texcoords(&mut self, texcoords: &[f32]) -> usize {
		let vertex_buffer = unsafe {
			asi_vulkan::new_buffer(
				&mut self.vw.connection,
				texcoords,
			)
		};

		let a = self.texcoords.len();

		self.texcoords.push(TexCoords {
			vertex_buffer,
			vertex_count: texcoords.len() as u32 / 4,
		});

		a
	}

	/// Push colors per vertex into graphics memory.
	pub fn colors(&mut self, colors: &[f32]) -> usize {
		let vertex_buffer = unsafe {
			asi_vulkan::new_buffer(
				&mut self.vw.connection,
				colors,
			)
		};

		let a = self.gradients.len();

		self.gradients.push(Gradient {
			vertex_buffer,
			vertex_count: colors.len() as u32 / 4,
		});

		a
	}

	pub fn textured(&mut self, model: usize, mat4: Mat4,
		texture: usize, texcoords: usize, alpha: bool,
		fog: bool, camera: bool) -> ShapeHandle
	{
		if self.models[model].vertex_count
			!= self.texcoords[texcoords].vertex_count
		{
			panic!("TexCoord length doesn't match vertex length");
		}

		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				if alpha {
					&self.style_texture
				} else {
					&self.style_natexture
				},
				TransformFullUniform {
					mat4: mat4.to_f32_array(),
					hcam: fog as u32 + camera as u32,
				},
				&self.camera_memory, // TODO: at shader creation, not shape creation
				Some(&self.effect_memory),
				Some(self.textures[texture].image.as_ref()
					.unwrap_or(&self.textures[texture]
						.mappable_image)),
				true, // 1 texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 2,
			buffers: [
				self.models[model].shape.buffer(),
				self.texcoords[texcoords].vertex_buffer.buffer(),
				unsafe { mem::uninitialized() }
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else if alpha {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		} else {
			ShapeHandle::Opaque(self.opaque_octree.add(shape))
		}
	}

	pub fn solid(&mut self, model: usize, mat4: Mat4, color: [f32; 4],
		alpha: bool, fog: bool, camera: bool)
		-> ShapeHandle
	{
		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				if alpha {
					&self.style_solid
				} else {
					&self.style_nasolid
				},
				TransformAndColorUniform {
					vec4: color,
					hcam: fog as u32 + camera as u32,
					mat4: mat4.to_f32_array(),
				},
				&self.camera_memory,
				Some(&self.effect_memory),
				None,
				false, // no texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 1,
			buffers: [
				self.models[model].shape.buffer(),
				unsafe { mem::uninitialized() },
				unsafe { mem::uninitialized() }
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else if alpha {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		} else {
			ShapeHandle::Opaque(self.opaque_octree.add(shape))
		}
	}

	pub fn gradient(&mut self, model: usize, mat4: Mat4, colors: usize,
		alpha: bool, fog: bool, camera: bool)
		-> ShapeHandle
	{
		if self.models[model].vertex_count
			!= self.gradients[colors].vertex_count
		{
			panic!("TexCoord length doesn't match gradient length");
		}

		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				if alpha {
					&self.style_gradient
				} else {
					&self.style_nagradient
				},
				TransformFullUniform {
					mat4: mat4.to_f32_array(),
					hcam: fog as u32 + camera as u32,
				},
				&self.camera_memory,
				Some(&self.effect_memory),
				None,
				false, // no texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 2,
			buffers: [
				self.models[model].shape.buffer(),
				self.gradients[colors].vertex_buffer.buffer(),
				unsafe { mem::uninitialized() }
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else if alpha {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		} else {
			ShapeHandle::Opaque(self.opaque_octree.add(shape))
		}
	}

	pub fn faded(&mut self, model: usize, mat4: Mat4, texture: usize,
		texcoords: usize, fade_factor: f32, fog: bool,
		camera: bool) -> ShapeHandle
	{
		if self.models[model].vertex_count
			!= self.texcoords[texcoords].vertex_count
		{
			panic!("TexCoord length doesn't match vertex length");
		}

		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				&self.style_faded,
				TransformAndFadeUniform {
					mat4: mat4.to_f32_array(),
					hcam: fog as u32 + camera as u32,
					fade: fade_factor,
				},
				&self.camera_memory,
				Some(&self.effect_memory),
				Some(self.textures[texture].image.as_ref()
					.unwrap_or(&self.textures[texture]
						.mappable_image)),
				true, // 1 texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 2,
			buffers: [
				self.models[model].shape.buffer(),
				self.texcoords[texcoords].vertex_buffer.buffer(),
				unsafe { mem::uninitialized() }
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		}
	}

	pub fn tinted(&mut self, model: usize, mat4: Mat4,
		texture: usize, texcoords: usize, color: [f32; 4],
		alpha: bool, fog: bool, camera: bool)
		-> ShapeHandle
	{
		if self.models[model].vertex_count
			!= self.texcoords[texcoords].vertex_count
		{
			panic!("TexCoord length doesn't match vertex length");
		}

		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				if alpha {
					&self.style_tinted
				} else {
					&self.style_natinted
				},
				TransformAndColorUniform {
					mat4: mat4.to_f32_array(),
					hcam: fog as u32 + camera as u32,
					vec4: color,
				},
				&self.camera_memory,
				Some(&self.effect_memory),
				Some(self.textures[texture].image.as_ref()
					.unwrap_or(&self.textures[texture]
						.mappable_image)),
				true, // 1 texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 2,
			buffers: [
				self.models[model].shape.buffer(),
				self.texcoords[texcoords].vertex_buffer.buffer(),
				unsafe { mem::uninitialized() }
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else if alpha {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		} else {
			ShapeHandle::Opaque(self.opaque_octree.add(shape))
		}
	}

	pub fn complex(&mut self, model: usize, mat4: Mat4,
		texture: usize, texcoords: usize, colors: usize, alpha: bool,
		fog: bool, camera: bool) -> ShapeHandle
	{
		if self.models[model].vertex_count
			!= self.texcoords[texcoords].vertex_count ||
			self.models[model].vertex_count
			!= self.gradients[colors].vertex_count
		{
			panic!("TexCoord length doesn't match vertex length");
		}

		let bbox = base::vertices_to_bbox(
			self.models[model].points.as_slice(), mat4
		);

		// Add an instance
		let instance = unsafe {
			Sprite::new(
				&mut self.vw.connection,
				if alpha {
					&self.style_complex
				} else {
					&self.style_nacomplex
				},
				TransformFullUniform {
					mat4: mat4.to_f32_array(),
					hcam: fog as u32 + camera as u32,
				},
				&self.camera_memory,
				Some(&self.effect_memory),
				Some(self.textures[texture].image.as_ref()
					.unwrap_or(&self.textures[texture]
						.mappable_image)),
				true, // 1 texure
			)
		};

		let shape = Shape {
			instance,
			num_buffers: 3,
			buffers: [
				self.models[model].shape.buffer(),
				self.texcoords[texcoords].vertex_buffer.buffer(),
				self.gradients[colors].vertex_buffer.buffer(),
			],
			model, bbox,
			fans: self.models[model].fans.clone(),
		};

		if !camera && !fog {
			self.gui_vec.push(shape);
			ShapeHandle::Gui(self.gui_vec.len() as u32 - 1)
		} else if alpha {
			ShapeHandle::Alpha(self.alpha_octree.add(shape))
		} else {
			ShapeHandle::Opaque(self.opaque_octree.add(shape))
		}
	}

	pub fn transform(&mut self, shape: &mut ShapeHandle, transform: ::Mat4){
		let uniform = TransformUniform {
			mat4: transform.to_f32_array(),
		};

		match shape {
			ShapeHandle::Opaque(ref mut x) => {
				let model = self.opaque_octree[*x].model;
				let bbox = base::vertices_to_bbox(
					self.models[model].points.as_slice(),
					transform);

				self.opaque_octree[*x].bbox = bbox;
				let shape = self.opaque_octree.remove(*x);
				*x = self.opaque_octree.add(shape);

				ffi::copy_memory(&mut self.vw.connection,
					self.opaque_octree[*x].instance.uniform_memory.memory(),
					&uniform);
			},
			ShapeHandle::Alpha(ref mut x) => {
				let model = self.alpha_octree[*x].model;
				let bbox = base::vertices_to_bbox(
					self.models[model].points.as_slice(),
					transform);

				self.alpha_octree[*x].bbox = bbox;
				let shape = self.alpha_octree.remove(*x);
				*x = self.alpha_octree.add(shape);

				ffi::copy_memory(&mut self.vw.connection,
					self.alpha_octree[*x].instance.uniform_memory.memory(),
					&uniform);
			},
			ShapeHandle::Gui(x) => {
				let x = *x as usize; // for indexing

				let model = self.gui_vec[x].model;
				let bbox = base::vertices_to_bbox(
					self.models[model].points.as_slice(),
					transform);
				self.gui_vec[x].bbox = bbox;

				ffi::copy_memory(&mut self.vw.connection,
					self.gui_vec[x].instance.uniform_memory.memory(),
					&uniform);
			},
		}
	}

	pub fn collision(&self, shape: &ShapeHandle, force: &mut ::ami::Vec3)
		-> Option<u32>
	{
		match shape {
			ShapeHandle::Opaque(x) => {
				self.opaque_octree.bounce_test(*x, force)
			},
			ShapeHandle::Alpha(x) => {
				self.alpha_octree.bounce_test(*x, force)
			},
			_ => panic!("No gui collision detection!")
		}
	}

	pub fn set_camera(&mut self, xyz: (f32,f32,f32), rxyz: (f32,f32,f32)) {
		self.xyz = xyz;
		self.rotate_xyz = rxyz;
	}

	pub fn camera(&mut self) {
		self.camera_memory.data.mat4 = (IDENTITY
			.translate(-self.xyz.0, -self.xyz.1, -self.xyz.2)
			.rotate(-self.rotate_xyz.0, -self.rotate_xyz.1,
				-self.rotate_xyz.2) * self.projection)
			.to_f32_array();

		self.camera_memory.update(&mut self.vw.connection);
	}

	pub fn fog(&mut self, fog: (f32, f32)) -> () {
		self.effect_memory.data.fogc = [self.clear_color.0,
			self.clear_color.1, self.clear_color.2, 1.0];
		self.effect_memory.data.fogr = [fog.0, fog.1];

		self.effect_memory.update(&mut self.vw.connection);
	}
}

impl Drop for Renderer {
	fn drop(&mut self) -> () {
		swapchain_delete(&mut self.vw);
	}
}
