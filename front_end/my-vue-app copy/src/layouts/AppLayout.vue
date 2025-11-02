<script setup>
import { computed } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const isAuthed = computed(() => !!localStorage.getItem("token"));

const logout = () => {
  localStorage.removeItem("token");
  router.push("/auth/login");
};
</script>

<template>
  <div>
    <header style="padding:1rem; border-bottom:1px solid #ccc; display:flex; gap:1rem;">
      <router-link to="/dashboard">Dashboard</router-link>
      <router-link to="/ideation">Ideation</router-link>
      <router-link to="/deliverables">Deliverables</router-link>

      <span style="margin-left:auto;"></span>
      <button v-if="isAuthed" @click="logout">Logout</button>
      <router-link v-else to="/auth/login">Login</router-link>
    </header>

    <main style="padding:1rem;">
      <router-view />
    </main>
  </div>
</template>
