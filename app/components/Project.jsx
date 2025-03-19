import { assets, workData } from '@/assets/assets'
import Image from 'next/image'
import React from 'react'

const Project = ({isDarkMode}) => {
  return (
    <div id="project" className='w-full px-[12%] py-10 scroll-mt-20'>
      <h2 className='text-center text-5xl font-Ovo'>Projects</h2>
      <p className='text-center max-w-2xl mx-auto mt-5 mb-12 font-Ovo'>
        My projects revolve around developing systems that combine the precision of robotics with the adapatability of AI, 
        enabling smarter homes, safer autonomous vehicles, and tools that amplify human potential.
      </p>


      <div className='grid grid-cols-auto my-10 gap-5 dark:text-black'>
              {workData.map((project, index) => (
                  <div key={index} className="flex flex-col gap-5 group cursor-pointer">
                  {/* Image Container */}
                  <div 
                      style={{ backgroundImage: `url(${project.bgImage})` }}
                      className="aspect-video bg-no-repeat bg-cover bg-center rounded-lg relative"
                  >
                  </div>
      
                  {/* Content Card */}
                  <div className='bg-white w-full rounded-md py-3 px-5 flex items-center justify-between 
                                  duration-500 transition-transform group-hover:-translate-y-2'>
                      <div>
                      <h2 className='font-semibold'>{project.title}</h2>
                      <p className='text-sm text-gray-700'>{project.description}</p>
                      </div>
                      <div className='border rounded-full border-black w-9 aspect-square flex items-center 
                                  justify-center shadow-[2px_2px_0_#000] group-hover:bg-lime-300 transition'>
                      <Image 
                          src={assets.send_icon} 
                          alt='send icon' 
                          className="w-5" 
                      />
                      </div>
                  </div>
                  </div>
              ))}
          </div>
          {/* <a href="" className='w-max flex items-center justify-center gap-2 text-gray-700 border-[0.5px] border-gray-700 rounded-full py-3 px-10 mx-auto
        my-20 hover:bg-lightHover duration-500 dark:text-white dark:border-white dark:hover:bg-darkHover'>
            Show more <Image src={isDarkMode ? assets.right_arrow_bold_dark : assets.right_arrow_bold} alt='Right arrow' className='w-4' />
        </a> */}

    </div>
    
  )
}

export default Project
