'use client'
import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Header from "../components/Header";
import Project_list from "../components/Project_list";

export default function ProjectPage() {

  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(()=>{
    if(localStorage.theme === 'dark' || (!(!'theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark').matches)){
      setIsDarkMode(true)
    }else{
      setIsDarkMode(false)
    }
  },[])

  useEffect(()=>{
    if(isDarkMode){
      document.documentElement.classList.add('dark')
      localStorage.theme = 'dark';
    }else{
      document.documentElement.classList.remove('dark')
      localStorage.theme = '';
    }
  },[isDarkMode])

  return (
    <>
      <Navbar isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />
      <Project_list isDarkMode={isDarkMode} />
      <Footer isDarkMode={isDarkMode} />
    </>
  );
}
