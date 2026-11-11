//! Events and textures other processes open.
//!
//! `newSharedEventHandle` and `newSharedTextureHandle` give an application an
//! object it may send to another process, which opens it with its own device.
//! Service handles are per connection, so what crosses is a [`ShareToken`]: 128
//! random bits the first sharer picks, a capability only as good as they are
//! unguessable. A shared object has a hold for each handle to it in any
//! connection and lives until the last goes. A token holds nothing, so the
//! frontend's handle object keeps its object alive in the task that shared it.
//! Every connection that ever held a handle may have committed work that
//! reads it, so its destruction waits for all of theirs.
use super::{
    Call, Kind, Object, Service,
    wire::{Fault, ShareObjectRequest, ShareToken, Status},
};
use crate::metal::{
    device::{
        EventId, TextureId,
        pins::{Reach, Scope},
    },
    shader::Compiler,
    vulkan::backend::Backend,
};
use alloc::collections::BTreeMap;

struct Share {
    token: [u64; 2],
    holds: u64,
    /// The connections that ever held a handle to it.
    holders: Reach,
}

/// What the service's connections share, by object and by token.
#[derive(Default)]
pub(super) struct Shares {
    objects: BTreeMap<Object, Share>,
    tokens: BTreeMap<[u64; 2], Object>,
}

impl Shares {
    /// Gives up one of several holds of `object`; whether it did, which leaves
    /// the object to the others.
    pub(super) fn release(&mut self, object: Object) -> bool {
        match self.objects.get_mut(&object) {
            Some(share) if share.holds > 1 => {
                share.holds -= 1;
                true
            }
            _ => false,
        }
    }

    /// Who may read `object` when the last hold goes, if it was shared:
    /// every command buffer of a connection that held it.
    pub(super) fn scope(&self, object: Object) -> Option<Scope> {
        self.objects.get(&object).map(|share| Scope {
            created: 0,
            reach: share.holders.clone(),
        })
    }

    /// `object` is gone, and its token with it.
    pub(super) fn forget(&mut self, object: Object) {
        if let Some(share) = self.objects.remove(&object) {
            self.tokens.remove(&share.token);
        }
    }

    /// The object of kind `T` that `token` names, with one more hold for a
    /// handle to it in `connection`.
    fn open<T: Kind>(&mut self, token: [u64; 2], connection: u64) -> Result<T, Fault> {
        let refused = Fault::new(Status::Invalid, "no such object has that share token");
        let object = *self.tokens.get(&token).ok_or(refused)?;
        let id = T::of(object).ok_or(refused)?;
        let share = self.objects.get_mut(&object).expect("a token's object");
        share.holds += 1;
        share.holders = share.holders.with(connection);
        Ok(id)
    }
}

impl<B: Backend, C: Compiler> Service<B, C> {
    /// Answers the object's token: the request's, unless it already has one.
    pub(super) fn share_object(&mut self, call: &mut Call) -> Result<(), Fault> {
        let ShareObjectRequest { handle, token } = call.request()?;
        let object = self.entry(call.connection, handle)?;
        if !matches!(object, Object::Event(_) | Object::Texture(_)) {
            return Err(Fault::new(
                Status::Invalid,
                "only events and textures are shared",
            ));
        }
        if let Object::Texture(texture) = object
            && self.device.is_staged_buffer_texture(texture)?
        {
            return Err(crate::metal::error::Error::Unsupported(
                "sharing host buffer render targets",
            )
            .into());
        }
        if let Some(share) = self.shares.objects.get(&object) {
            let token = share.token;
            return call.reply(&ShareToken { token });
        }
        if self.shares.tokens.contains_key(&token) {
            return Err(Fault::new(Status::Invalid, "a share token in use"));
        }
        self.shares.tokens.insert(token, object);
        let holds = 1;
        let holders = Reach::One(call.connection.get());
        self.shares.objects.insert(
            object,
            Share {
                token,
                holds,
                holders,
            },
        );
        call.reply(&ShareToken { token })
    }

    pub(super) fn open_shared_event(&mut self, call: &mut Call) -> Result<(), Fault> {
        let ShareToken { token } = call.request()?;
        let event: EventId = self.shares.open(token, call.connection.get())?;
        let handle = self.hold(call.connection, Object::Event(event))?;
        call.reply(&handle)
    }

    pub(super) fn open_shared_texture(&mut self, call: &mut Call) -> Result<(), Fault> {
        let ShareToken { token } = call.request()?;
        let texture: TextureId = self.shares.open(token, call.connection.get())?;
        let response = self.adopt_texture(call.connection, texture)?;
        call.reply(&response)
    }
}
